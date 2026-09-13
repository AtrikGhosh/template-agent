"""Unit tests for runtime OAuth/DCR MCP tool attach middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deep_agent.aegra.mcp_runtime_tools import (
    McpRuntimeToolsMiddleware,
    _apply_live_hitl_decisions,
    _is_auth_continue,
)


def _model_request(tools: list | None = None):
    req = MagicMock()
    req.tools = tools or []

    def _override(**kwargs):
        new_req = MagicMock()
        new_req.tools = kwargs.get("tools", req.tools)
        new_req.override = _override
        return new_req

    req.override = _override
    return req


def _tool_request(*, name: str, tool=None, call_id: str = "call_1", args=None):
    req = MagicMock()
    req.tool = tool
    req.tool_call = {"name": name, "id": call_id, "args": args or {}}

    def _override(**kwargs):
        new_req = MagicMock()
        new_req.tool = kwargs.get("tool", req.tool)
        new_req.tool_call = kwargs.get("tool_call", req.tool_call)
        new_req.override = _override
        return new_req

    req.override = _override
    return req


def _ai_state(*calls: dict):
    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


_SERVERS = {
    "jira-mcp": {"enabled": True, "auth_mode": "dcr", "tool_prefix": "jira"},
    "template-mcp-server": {
        "enabled": True,
        "auth_mode": "sso",
        "tool_prefix": "template",
    },
}


class TestAwrapModelCall:
    @pytest.mark.asyncio
    async def test_adds_live_tools_when_placeholder_and_token(self):
        placeholder = MagicMock()
        placeholder.name = "mcp__jira_mcp"
        live = MagicMock()
        live.name = "jira_search"
        req = _model_request([placeholder])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            mock_ctx.set = MagicMock()
            result = await mw.awrap_model_call(req, handler)
        assert result == "ok"
        overridden = handler.call_args[0][0]
        names = [t.name for t in overridden.tools]
        assert "jira_search" in names
        assert "mcp__jira_mcp" in names

    @pytest.mark.asyncio
    async def test_skips_when_no_user(self):
        req = _model_request()
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with patch("deep_agent.aegra.mcp._resolve_mcp_user_id", return_value=None):
            result = await mw.awrap_model_call(req, handler)
        assert result == "ok"
        handler.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_does_not_duplicate_already_present_tools(self):
        placeholder = MagicMock()
        placeholder.name = "mcp__jira_mcp"
        live = MagicMock()
        live.name = "jira_search"
        req = _model_request([placeholder, live])
        handler = AsyncMock(return_value="ok")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value={
                    "jira-mcp": {"enabled": True, "auth_mode": "dcr"},
                },
            ),
        ):
            await mw.awrap_model_call(req, handler)
        handler.assert_awaited_once_with(req)


class TestAwrapToolCall:
    @pytest.mark.asyncio
    async def test_compiled_tool_passes_through(self):
        compiled = MagicMock()
        req = _tool_request(name="jira_search", tool=compiled)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        result = await mw.awrap_tool_call(req, handler)
        assert result == "ran"
        handler.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_overrides_unregistered_tool_without_hitl(self):
        live = MagicMock()
        live.name = "jira_search"
        req = _tool_request(name="jira_search", tool=None)
        handler = AsyncMock(return_value="ran")
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                new=AsyncMock(return_value=[live]),
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            result = await mw.awrap_tool_call(req, handler)
        assert result == "ran"
        mock_interrupt.assert_not_called()
        overridden = handler.call_args[0][0]
        assert overridden.tool is live


class TestAafterModel:
    @pytest.mark.asyncio
    async def test_auth_interrupt_when_token_missing(self):
        state = _ai_state(
            {
                "name": "mcp__jira_mcp",
                "id": "c1",
                "args": {"query": ""},
            }
        )
        mw = McpRuntimeToolsMiddleware()
        resolve = AsyncMock(side_effect=[None, "tok"])
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=resolve,
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            ) as mock_resolver,
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value="continue",
            ) as mock_interrupt,
        ):
            mock_resolver.return_value.connect_url.return_value = (
                "/mcp/jira-mcp/connect"
            )
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "mcp_auth_required" in payload
        assert "jira-mcp" in payload

    @pytest.mark.asyncio
    async def test_hitl_for_live_tool_when_token_present(self):
        state = _ai_state(
            {
                "name": "template_validate_email",
                "id": "e1",
                "args": {"email": "pat@redhat.com"},
            },
            {
                "name": "jira_search",
                "id": "j1",
                "args": {"q": "bugs"},
            },
        )
        hitl = MagicMock()
        hitl.enabled = True
        hitl.mode = "all"
        hitl.exclude = []
        resolved = MagicMock()
        resolved.human_approval = hitl
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=resolved,
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "approve"}]},
            ) as mock_interrupt,
        ):
            result = await mw.aafter_model(state, None)
        assert result is not None
        payload = mock_interrupt.call_args[0][0]
        assert payload["action_requests"][0]["name"] == "jira_search"
        names = [c["name"] for c in result["messages"][0].tool_calls]
        assert "template_validate_email" in names
        assert "jira_search" in names

    @pytest.mark.asyncio
    async def test_reject_drops_live_call(self):
        state = _ai_state(
            {
                "name": "jira_search",
                "id": "j1",
                "args": {"q": "bugs"},
            }
        )
        hitl = MagicMock()
        hitl.enabled = True
        hitl.mode = "all"
        hitl.exclude = []
        resolved = MagicMock()
        resolved.human_approval = hitl
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp._resolve_connection_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.get_orchestrator_config",
                return_value={"model": "gemini-x"},
            ),
            patch(
                "deep_agent.src.agent.config.agent_config.resolve_agent_middleware",
                return_value=resolved,
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
                return_value={"decisions": [{"type": "reject", "message": "nope"}]},
            ),
        ):
            result = await mw.aafter_model(state, None)
        assert result is not None
        assert result["messages"][0].tool_calls == []
        rejected = result["messages"][1]
        assert isinstance(rejected, ToolMessage)
        assert rejected.status == "error"
        assert "nope" in rejected.content

    @pytest.mark.asyncio
    async def test_sso_only_does_not_interrupt(self):
        state = _ai_state(
            {
                "name": "template_validate_email",
                "id": "e1",
                "args": {"email": "a@b.com"},
            }
        )
        mw = McpRuntimeToolsMiddleware()
        with (
            patch(
                "deep_agent.aegra.mcp._resolve_mcp_user_id",
                return_value="user-1",
            ),
            patch("deep_agent.aegra.mcp._current_user_id"),
            patch(
                "deep_agent.aegra.mcp._get_server_configs",
                return_value=_SERVERS,
            ),
            patch(
                "deep_agent.aegra.mcp_runtime_tools.interrupt",
            ) as mock_interrupt,
        ):
            result = await mw.aafter_model(state, None)
        assert result is None
        mock_interrupt.assert_not_called()


class TestApplyLiveHitlDecisions:
    def test_reject_returns_error_tool_message(self):
        calls = [{"name": "jira_search", "id": "j1", "args": {}}]
        revised, messages = _apply_live_hitl_decisions(
            calls, [0], {"decisions": [{"type": "reject", "message": "nope"}]}
        )
        assert revised == []
        assert messages[0].status == "error"
        assert "nope" in messages[0].content

    def test_continue_is_detected(self):
        assert _is_auth_continue("continue")
        assert _is_auth_continue({"type": "continue"})
        assert not _is_auth_continue({"decisions": [{"type": "approve"}]})


class TestRewriteThenRuntimeAttach:
    @pytest.mark.asyncio
    async def test_live_frontmatter_name_attaches_after_rewrite(self):
        from deep_agent.aegra.mcp import rewrite_oauth_dcr_tool_names
        from deep_agent.src.agent.config.resolver import resolve_tools

        placeholder = MagicMock()
        placeholder.name = "mcp__jira_mcp"
        live = MagicMock()
        live.name = "jira_search"
        servers = {
            "jira-mcp": {
                "enabled": True,
                "auth_mode": "dcr",
                "tool_prefix": "jira",
            }
        }
        with patch(
            "deep_agent.aegra.mcp._get_server_configs",
            return_value=servers,
        ):
            names = rewrite_oauth_dcr_tool_names(["jira_search"])
            bound = resolve_tools(names, [placeholder], "jira-child")
            req = _model_request(bound)
            handler = AsyncMock(return_value="ok")
            mw = McpRuntimeToolsMiddleware()
            with (
                patch(
                    "deep_agent.aegra.mcp._resolve_mcp_user_id",
                    return_value="user-1",
                ),
                patch("deep_agent.aegra.mcp._current_user_id") as mock_ctx,
                patch(
                    "deep_agent.aegra.mcp.get_authenticated_oauth_mcp_tools",
                    new=AsyncMock(return_value=[live]),
                ),
            ):
                mock_ctx.set = MagicMock()
                result = await mw.awrap_model_call(req, handler)
        assert result == "ok"
        overridden = handler.call_args[0][0]
        names_out = [t.name for t in overridden.tools]
        assert "mcp__jira_mcp" in names_out
        assert "jira_search" in names_out
