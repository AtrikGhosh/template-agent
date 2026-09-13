"""Runtime attach of OAuth/DCR MCP tools after Connect, without rebuilding the graph.

Compile-time graphs bind a stable placeholder per oauth/dcr server. After the
user token is in Redis, this middleware lists the real tools and:

- ``awrap_model_call``: shows those tools to the model
- ``aafter_model``: Authenticate / Approve **before** any tool in the batch runs
- ``awrap_tool_call``: executes names that were not on the compiled ToolNode

HITL ``mode: all`` does not include runtime names in ``interrupt_on``. Those
calls are paused in ``aafter_model`` with the same HITL payload the UI already
understands, so an in-tools-node interrupt cannot replay sibling MCP calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_HITL_DECISIONS = ["approve", "edit", "reject", "respond"]


def _tool_call_name_and_id(tool_call: Any) -> tuple[str, str]:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or ""), str(tool_call.get("id") or "")
    return str(getattr(tool_call, "name", "") or ""), str(
        getattr(tool_call, "id", "") or ""
    )


def _tool_call_args(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        args = tool_call.get("args") or {}
    else:
        args = getattr(tool_call, "args", {}) or {}
    return args if isinstance(args, dict) else {}


def _is_auth_continue(raw: Any) -> bool:
    return raw in (None, "continue") or raw == {"type": "continue"}


def _runtime_hitl_required(tool_name: str) -> bool:
    """Whether a dynamically attached MCP tool must pause for human approval."""
    try:
        from deep_agent.src.agent.config import agent_config

        orch = agent_config.get_orchestrator_config()
        resolved = agent_config.resolve_agent_middleware(orch.get("model") or "")
        hitl = resolved.human_approval
    except Exception:
        logger.warning(
            "Could not read HITL config for runtime MCP tool '%s' — pausing",
            tool_name,
            exc_info=True,
        )
        return True
    if not hitl.enabled or hitl.mode == "none":
        return False
    if tool_name in hitl.exclude:
        return False
    return hitl.mode == "all"


def _is_live_oauth_dcr_name(name: str) -> bool:
    from deep_agent.aegra.mcp import oauth_dcr_server_for_tool_name

    if not name or name.startswith("mcp__"):
        return False
    return oauth_dcr_server_for_tool_name(name) is not None


def _hitl_payload_for_calls(calls: list[Any]) -> dict[str, Any]:
    action_requests = []
    review_configs = []
    for call in calls:
        name, _ = _tool_call_name_and_id(call)
        action_requests.append(
            {
                "name": name,
                "args": _tool_call_args(call),
                "description": f"Approve MCP tool '{name}'",
            }
        )
        review_configs.append(
            {
                "action_name": name,
                "allowed_decisions": list(_HITL_DECISIONS),
            }
        )
    return {"action_requests": action_requests, "review_configs": review_configs}


def _interrupt_hitl(payload: dict[str, Any]) -> Any:
    """Pause for Approve. Auth-style ``continue`` is not an Approve."""
    raw = interrupt(payload)
    if _is_auth_continue(raw):
        raw = interrupt(payload)
    return raw


def _decision_at(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"type": "approve"}
    decisions = raw.get("decisions") or []
    decision = decisions[index] if index < len(decisions) else {"type": "approve"}
    return decision if isinstance(decision, dict) else {"type": "approve"}


def _apply_live_hitl_decisions(
    tool_calls: list[Any],
    live_indices: list[int],
    raw: Any,
) -> tuple[list[Any], list[ToolMessage]]:
    """Keep approved/edited live calls; rejected/responded calls become ToolMessages."""
    live_set = set(live_indices)
    revised: list[Any] = []
    messages: list[ToolMessage] = []
    decision_idx = 0
    for idx, call in enumerate(tool_calls):
        if idx not in live_set:
            revised.append(call)
            continue
        name, tool_call_id = _tool_call_name_and_id(call)
        decision = _decision_at(raw, decision_idx)
        decision_idx += 1
        kind = decision.get("type") or "approve"
        if kind == "reject":
            reason = str(decision.get("message") or "") or (
                "The user rejected this tool call."
            )
            messages.append(
                ToolMessage(
                    content=reason,
                    name=name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            )
            continue
        if kind == "respond":
            messages.append(
                ToolMessage(
                    content=str(decision.get("message") or ""),
                    name=name,
                    tool_call_id=tool_call_id,
                    status="success",
                )
            )
            continue
        if kind == "edit":
            edited = decision.get("edited_action") or {}
            new_call = (
                dict(call)
                if isinstance(call, dict)
                else {
                    "name": name,
                    "args": _tool_call_args(call),
                    "id": tool_call_id,
                }
            )
            if isinstance(edited, dict):
                if edited.get("name"):
                    new_call["name"] = edited["name"]
                if "args" in edited:
                    new_call["args"] = edited["args"]
            revised.append(new_call)
            continue
        revised.append(call)
    return revised, messages


class McpRuntimeToolsMiddleware(AgentMiddleware):
    """Attach authenticated OAuth/DCR MCP tools at call time (stable compiled graph)."""

    name = "McpRuntimeToolsMiddleware"

    def _placeholder_server_names(self, tools: list[Any]) -> list[str]:
        from deep_agent.aegra.mcp import _get_server_configs, placeholder_tool_name

        bound = {getattr(t, "name", "") for t in tools}
        names: list[str] = []
        for key, cfg in _get_server_configs().items():
            if not cfg.get("enabled", False):
                continue
            if cfg.get("auth_mode") not in ("oauth", "dcr"):
                continue
            if placeholder_tool_name(key) in bound:
                names.append(key)
        return names

    async def _interrupt_missing_oauth_tokens(
        self, user_id: str, server_keys: list[str]
    ) -> None:
        from deep_agent.aegra.mcp import _get_server_configs, _resolve_connection_token
        from deep_agent.aegra.mcp_auth import (
            NeedsAuthorization,
            get_mcp_credential_resolver,
        )
        from deep_agent.aegra.mcp_tool_auth import _mcp_auth_interrupt_payload

        configs = _get_server_configs()
        seen: set[str] = set()
        ordered: list[str] = []
        for key in server_keys:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        while True:
            missing: str | None = None
            for key in ordered:
                entry = configs.get(key)
                if entry is None:
                    continue
                token = await _resolve_connection_token(key, entry, None, user_id)
                if not token:
                    missing = key
                    break
            if missing is None:
                return
            exc = NeedsAuthorization(
                missing,
                get_mcp_credential_resolver().connect_url(missing),
            )
            interrupt(_mcp_auth_interrupt_payload(exc))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Attach live OAuth/DCR tools to the model request when a token exists."""
        from deep_agent.aegra.mcp import (
            _current_user_id,
            _resolve_mcp_user_id,
            get_authenticated_oauth_mcp_tools,
        )

        user_id = _resolve_mcp_user_id()
        if user_id:
            _current_user_id.set(user_id)
        if not user_id:
            return await handler(request)

        server_names = self._placeholder_server_names(list(request.tools or []))
        if not server_names:
            return await handler(request)

        try:
            live = await get_authenticated_oauth_mcp_tools(
                user_id, server_names=server_names
            )
        except Exception:
            logger.warning(
                "Authenticated MCP tool listing failed — continuing without runtime tools",
                exc_info=True,
            )
            return await handler(request)

        existing = {getattr(t, "name", "") for t in request.tools or []}
        extra = [t for t in live if getattr(t, "name", "") not in existing]
        if not extra:
            return await handler(request)
        return await handler(request.override(tools=[*request.tools, *extra]))

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Pause for Connect and HITL before any live OAuth/DCR tool in the batch runs."""
        from deep_agent.aegra.mcp import (
            _current_user_id,
            _resolve_mcp_user_id,
            oauth_dcr_server_for_tool_name,
        )

        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return None
        last_ai = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if last_ai is None or not last_ai.tool_calls:
            return None

        tool_calls = list(last_ai.tool_calls)
        server_keys: list[str] = []
        live_indices: list[int] = []
        for idx, call in enumerate(tool_calls):
            name, _ = _tool_call_name_and_id(call)
            key = oauth_dcr_server_for_tool_name(name)
            if key:
                server_keys.append(key)
            if _is_live_oauth_dcr_name(name) and _runtime_hitl_required(name):
                live_indices.append(idx)

        user_id = _resolve_mcp_user_id()
        if user_id:
            _current_user_id.set(user_id)
            if server_keys:
                await self._interrupt_missing_oauth_tokens(user_id, server_keys)

        if not live_indices:
            return None

        payload = _hitl_payload_for_calls([tool_calls[i] for i in live_indices])
        raw = _interrupt_hitl(payload)
        revised, artificial = _apply_live_hitl_decisions(tool_calls, live_indices, raw)
        last_ai.tool_calls = revised
        if not artificial:
            return {"messages": [last_ai]}
        return {"messages": [last_ai, *artificial]}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Execute a live OAuth/DCR tool that was not bound on the compiled ToolNode."""
        from deep_agent.aegra.mcp import (
            _current_user_id,
            _resolve_mcp_user_id,
            get_authenticated_oauth_mcp_tools,
        )

        name, _tool_call_id = _tool_call_name_and_id(request.tool_call)
        if request.tool is not None:
            return await handler(request)

        user_id = _resolve_mcp_user_id()
        if user_id:
            _current_user_id.set(user_id)
        if not user_id:
            return await handler(request)

        try:
            live = await get_authenticated_oauth_mcp_tools(user_id)
        except Exception:
            logger.warning(
                "Authenticated MCP tool lookup failed for '%s'",
                name,
                exc_info=True,
            )
            return await handler(request)

        live_tool = next(
            (t for t in live if getattr(t, "name", "") == name),
            None,
        )
        if live_tool is None:
            return await handler(request)
        return await handler(request.override(tool=live_tool))


def build_mcp_runtime_tools_middleware() -> McpRuntimeToolsMiddleware:
    """Factory used by the orchestrator, subagents, and harness extra_middleware."""
    return McpRuntimeToolsMiddleware()
