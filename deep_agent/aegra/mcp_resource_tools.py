"""Host-side LangChain tools that call MCP ``resources/*`` (not server tools).

The model cannot speak JSON-RPC. These tools are the host adapter: they open the
same request-scoped MCP session as the Apps HTTP proxy, then return the payload
unchanged. No catalog cache — safe for multi-pod.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field as PydanticField

from deep_agent.aegra.mcp import (
    _current_access_token,
    _current_user_id,
    _filter_by_names,
    _get_server_configs,
)
from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.aegra.mcp_host import (
    list_resource_templates,
    list_resources,
    read_resource,
)

LIST_TOOL = "mcp_list_resources"
TEMPLATES_TOOL = "mcp_list_resource_templates"
READ_TOOL = "mcp_read_resource"


def get_mcp_resource_tools(
    *,
    server_names: list[str] | None = None,
    allowed_uris: list[str] | None = None,
) -> list[Any]:
    """Return host resource tools scoped like ``get_mcp_tools``.

    Filters enabled MCP servers (and optional ``mcps:`` names) with the same
    helpers as tool discovery. Does **not** connect or call ``resources/list``
    — the three tools talk to the server at turn time.
    """
    servers = _get_server_configs()
    enabled = {
        k: v
        for k, v in servers.items()
        if isinstance(v, dict) and v.get("enabled", False)
    }
    enabled = _filter_by_names(enabled, server_names)
    return build_mcp_resource_tools(
        allowed_servers=list(enabled.keys()),
        allowed_uris=allowed_uris,
    )


def _uri_allowed(uri: str, allowed_uris: list[str] | None) -> bool:
    """Return True if *uri* is unrestricted, listed, or matches a listed template."""
    if allowed_uris is None:
        return True
    if uri in allowed_uris:
        return True
    return any(
        "{" in pattern and _template_matches(pattern, uri) for pattern in allowed_uris
    )


def _template_matches(pattern: str, uri: str) -> bool:
    """Match RFC-6570-style ``{param}`` as a single URI path segment."""
    parts = re.split(r"(\{[^}]+\})", pattern)
    regex = (
        "^"
        + "".join(
            r"[^/]+" if p.startswith("{") and p.endswith("}") else re.escape(p)
            for p in parts
        )
        + "$"
    )
    return re.match(regex, uri) is not None


def _auth_context() -> tuple[str, str | None]:
    return _current_user_id.get() or "", _current_access_token.get()


def _is_authorization_required(exc: HTTPException) -> bool:
    detail = exc.detail
    return (
        exc.status_code == 401
        and isinstance(detail, dict)
        and detail.get("error") == "authorization_required"
        and isinstance(detail.get("mcp_name"), str)
        and isinstance(detail.get("connect_url"), str)
    )


def _raise_or_format_http(exc: HTTPException) -> str:
    if _is_authorization_required(exc):
        detail = exc.detail
        raise NeedsAuthorization(detail["mcp_name"], detail["connect_url"]) from None
    return f"MCP resource request failed ({exc.status_code}): {exc.detail}"


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _filter_resources(
    payload: dict[str, Any], allowed_uris: list[str] | None
) -> dict[str, Any]:
    if allowed_uris is None:
        return payload
    out = dict(payload)
    out["resources"] = [
        item
        for item in (payload.get("resources") or [])
        if isinstance(item, dict)
        and _uri_allowed(str(item.get("uri") or ""), allowed_uris)
    ]
    return out


def _filter_templates(
    payload: dict[str, Any], allowed_uris: list[str] | None
) -> dict[str, Any]:
    if allowed_uris is None:
        return payload
    key = (
        "resourceTemplates" if "resourceTemplates" in payload else "resource_templates"
    )
    out = dict(payload)
    out[key] = [
        item
        for item in (payload.get(key) or [])
        if isinstance(item, dict)
        and str(item.get("uriTemplate") or item.get("uri_template") or "")
        in allowed_uris
    ]
    return out


def _reject_server(mcp_name: str, allowed_servers: tuple[str, ...]) -> str:
    allowed = ", ".join(allowed_servers) if allowed_servers else "(none)"
    return f"Unknown or disallowed MCP server {mcp_name!r}. Allowed: {allowed}"


def build_mcp_resource_tools(
    *,
    allowed_servers: list[str],
    allowed_uris: list[str] | None = None,
) -> list[Any]:
    """Return list/templates/read tools, or ``[]`` when nothing is allowed.

    Args:
        allowed_servers: MCP ``mcp.json`` keys this agent may call.
        allowed_uris: ``None`` = all URIs; ``[]`` = no tools; otherwise allowlist
            of concrete URIs and ``uriTemplate`` strings.
    """
    servers = tuple(s for s in allowed_servers if s)
    if not servers or allowed_uris == []:
        return []

    server_list = ", ".join(servers)

    class _ListInput(BaseModel):
        mcp_name: str = PydanticField(
            description=f"MCP server name. Allowed: {server_list}"
        )
        cursor: str | None = PydanticField(
            default=None,
            description="Pagination cursor from a previous list call",
        )

    class _ReadInput(BaseModel):
        mcp_name: str = PydanticField(
            description=f"MCP server name. Allowed: {server_list}"
        )
        uri: str = PydanticField(
            description="Resource URI from resources/list or a template"
        )

    async def _list(mcp_name: str, cursor: str | None = None) -> str:
        if mcp_name not in servers:
            return _reject_server(mcp_name, servers)
        user_id, sso = _auth_context()
        try:
            payload = await list_resources(
                mcp_name, cursor=cursor, user_id=user_id, sso_token=sso
            )
        except NeedsAuthorization:
            raise
        except HTTPException as exc:
            return _raise_or_format_http(exc)
        except Exception as exc:
            return f"MCP resource request failed: {exc}"
        return _dump(_filter_resources(payload, allowed_uris))

    async def _templates(mcp_name: str, cursor: str | None = None) -> str:
        if mcp_name not in servers:
            return _reject_server(mcp_name, servers)
        user_id, sso = _auth_context()
        try:
            payload = await list_resource_templates(
                mcp_name, cursor=cursor, user_id=user_id, sso_token=sso
            )
        except NeedsAuthorization:
            raise
        except HTTPException as exc:
            return _raise_or_format_http(exc)
        except Exception as exc:
            return f"MCP resource request failed: {exc}"
        return _dump(_filter_templates(payload, allowed_uris))

    async def _read(mcp_name: str, uri: str) -> str:
        if mcp_name not in servers:
            return _reject_server(mcp_name, servers)
        if not _uri_allowed(uri, allowed_uris):
            return f"Resource URI not allowed: {uri}"
        user_id, sso = _auth_context()
        try:
            payload = await read_resource(mcp_name, uri, user_id=user_id, sso_token=sso)
        except NeedsAuthorization:
            raise
        except HTTPException as exc:
            return _raise_or_format_http(exc)
        except Exception as exc:
            return f"MCP resource request failed: {exc}"
        return _dump(payload)

    list_desc = (
        "List MCP resources (resources/list). Returns catalog metadata, not file "
        "contents. If the list is empty, call mcp_list_resource_templates. "
        f"mcp_name must be one of: {server_list}."
    )
    templates_desc = (
        "List MCP resource templates (resources/templates/list) for parameterized "
        f"URIs. mcp_name must be one of: {server_list}."
    )
    read_desc = (
        "Read an MCP resource (resources/read) by URI. Use after listing. "
        f"mcp_name must be one of: {server_list}."
    )

    return [
        StructuredTool(
            name=LIST_TOOL,
            description=list_desc,
            func=lambda **_: "",
            coroutine=_list,
            args_schema=_ListInput,
        ),
        StructuredTool(
            name=TEMPLATES_TOOL,
            description=templates_desc,
            func=lambda **_: "",
            coroutine=_templates,
            args_schema=_ListInput,
        ),
        StructuredTool(
            name=READ_TOOL,
            description=read_desc,
            func=lambda **_: "",
            coroutine=_read,
            args_schema=_ReadInput,
        ),
    ]
