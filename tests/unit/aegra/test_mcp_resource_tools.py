"""Unit tests for host MCP resource tools (resources/list, templates, read)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from deep_agent.aegra.mcp import _current_access_token, _current_user_id
from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.aegra.mcp_resource_tools import (
    BLOB_OMITTED,
    LIST_TOOL,
    READ_TOOL,
    TEMPLATES_TOOL,
    build_mcp_resource_tools,
)


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


@pytest.fixture
def auth_ctx():
    uid = _current_user_id.set("user-1")
    tok = _current_access_token.set("sso-token")
    try:
        yield
    finally:
        _current_user_id.reset(uid)
        _current_access_token.reset(tok)


class TestGetMcpResourceTools:
    def test_all_enabled_when_server_names_omitted(self):
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={
                    "a": {"enabled": True},
                    "b": {"enabled": False},
                    "c": {"enabled": True},
                },
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
                return_value=[],
            ) as mock_build,
        ):
            from deep_agent.aegra.mcp_resource_tools import get_mcp_resource_tools

            get_mcp_resource_tools(server_names=None)
        mock_build.assert_called_once_with(
            allowed_servers=["a", "c"], allowed_uris=None
        )

    def test_intersects_declared_mcps_with_enabled(self):
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={
                    "a": {"enabled": True},
                    "b": {"enabled": False},
                    "c": {"enabled": True},
                },
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
                return_value=[],
            ) as mock_build,
        ):
            from deep_agent.aegra.mcp_resource_tools import get_mcp_resource_tools

            get_mcp_resource_tools(server_names=["b", "c", "missing"])
        mock_build.assert_called_once_with(allowed_servers=["c"], allowed_uris=None)

    def test_forwards_allowed_uris(self):
        with (
            patch(
                "deep_agent.aegra.mcp_resource_tools._get_server_configs",
                return_value={"a": {"enabled": True}},
            ),
            patch(
                "deep_agent.aegra.mcp_resource_tools.build_mcp_resource_tools",
                return_value=[],
            ) as mock_build,
        ):
            from deep_agent.aegra.mcp_resource_tools import get_mcp_resource_tools

            get_mcp_resource_tools(server_names=None, allowed_uris=["template://about"])
        mock_build.assert_called_once_with(
            allowed_servers=["a"], allowed_uris=["template://about"]
        )


class TestBuildMcpResourceTools:
    def test_empty_uri_allowlist_returns_no_tools(self):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=[],
        )
        assert tools == []

    def test_no_servers_returns_no_tools(self):
        tools = build_mcp_resource_tools(allowed_servers=[], allowed_uris=None)
        assert tools == []

    def test_unrestricted_returns_three_tools(self):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        assert [t.name for t in tools] == [LIST_TOOL, TEMPLATES_TOOL, READ_TOOL]
        assert "omitted" in _tool(tools, READ_TOOL).description


class TestListResourcesTool:
    @pytest.mark.asyncio
    async def test_rejects_unknown_server(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "other"})
        assert "disallowed" in result
        assert "template-mcp-server" in result

    @pytest.mark.asyncio
    async def test_passes_cursor_and_auth(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["template-mcp-server"],
            allowed_uris=None,
        )
        payload = {
            "resources": [{"uri": "template://about", "name": "about"}],
            "nextCursor": "page-2",
        }
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            return_value=payload,
        ) as mock_list:
            result = await _tool(tools, LIST_TOOL).ainvoke(
                {"mcp_name": "template-mcp-server", "cursor": "abc"}
            )
        mock_list.assert_awaited_once_with(
            "template-mcp-server",
            cursor="abc",
            user_id="user-1",
            sso_token="sso-token",
        )
        parsed = json.loads(result)
        assert parsed["resources"][0]["uri"] == "template://about"
        assert parsed["nextCursor"] == "page-2"

    @pytest.mark.asyncio
    async def test_filters_list_by_allowlist(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://about"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            return_value={
                "resources": [
                    {"uri": "template://about", "name": "about"},
                    {"uri": "template://secret", "name": "secret"},
                ]
            },
        ):
            result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        uris = [r["uri"] for r in json.loads(result)["resources"]]
        assert uris == ["template://about"]

    @pytest.mark.asyncio
    async def test_authorization_required_raises(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            side_effect=HTTPException(
                status_code=401,
                detail={
                    "error": "authorization_required",
                    "mcp_name": "s",
                    "connect_url": "/mcp/s/connect",
                },
            ),
        ):
            with pytest.raises(NeedsAuthorization) as exc:
                await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        assert exc.value.mcp_name == "s"
        assert exc.value.connect_url == "/mcp/s/connect"

    @pytest.mark.asyncio
    async def test_other_http_errors_are_strings(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resources",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=404, detail="Unknown or disabled"),
        ):
            result = await _tool(tools, LIST_TOOL).ainvoke({"mcp_name": "s"})
        assert result.startswith("MCP resource request failed (404)")


class TestListTemplatesTool:
    @pytest.mark.asyncio
    async def test_filters_templates_by_exact_uri_template(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://echo/{text}"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.list_resource_templates",
            new_callable=AsyncMock,
            return_value={
                "resourceTemplates": [
                    {"uriTemplate": "template://echo/{text}", "name": "echo"},
                    {"uriTemplate": "template://other/{id}", "name": "other"},
                ]
            },
        ):
            result = await _tool(tools, TEMPLATES_TOOL).ainvoke({"mcp_name": "s"})
        templates = json.loads(result)["resourceTemplates"]
        assert len(templates) == 1
        assert templates[0]["uriTemplate"] == "template://echo/{text}"


class TestReadResourceTool:
    @pytest.mark.asyncio
    async def test_keeps_text_and_stubs_blob(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        blob = "cG5nLWJ5dGVz"
        payload = {
            "contents": [
                {"uri": "template://about", "mimeType": "text/plain", "text": "hello"},
                {"uri": "template://logo", "mimeType": "image/png", "blob": blob},
            ]
        }
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value=payload,
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://logo"}
            )
        mock_read.assert_awaited_once_with(
            "s", "template://logo", user_id="user-1", sso_token="sso-token"
        )
        parsed = json.loads(result)
        assert parsed["contents"][0]["text"] == "hello"
        logo = parsed["contents"][1]
        assert "blob" not in logo
        assert logo["text"] == BLOB_OMITTED
        assert blob not in result

    @pytest.mark.asyncio
    async def test_text_only_is_unchanged(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        payload = {
            "contents": [
                {"uri": "template://about", "mimeType": "text/plain", "text": "hello"}
            ]
        }
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value=payload,
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://about"}
            )
        assert json.loads(result) == payload

    @pytest.mark.asyncio
    async def test_keeps_text_when_same_item_has_blob(self, auth_ctx):
        tools = build_mcp_resource_tools(allowed_servers=["s"], allowed_uris=None)
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={
                "contents": [
                    {
                        "uri": "template://both",
                        "mimeType": "text/plain",
                        "text": "caption",
                        "blob": "eA==",
                    }
                ]
            },
        ):
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://both"}
            )
        item = json.loads(result)["contents"][0]
        assert item["text"] == "caption"
        assert "blob" not in item

    @pytest.mark.asyncio
    async def test_rejects_uri_not_on_allowlist(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://about"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://secret"}
            )
        mock_read.assert_not_awaited()
        assert "not allowed" in result

    @pytest.mark.asyncio
    async def test_allows_uri_matching_listed_template(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://echo/{text}"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
            return_value={"contents": [{"uri": "template://echo/hi", "text": "hi"}]},
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://echo/hi"}
            )
        mock_read.assert_awaited_once()
        assert json.loads(result)["contents"][0]["text"] == "hi"

    @pytest.mark.asyncio
    async def test_template_does_not_match_extra_path_segment(self, auth_ctx):
        tools = build_mcp_resource_tools(
            allowed_servers=["s"],
            allowed_uris=["template://echo/{text}"],
        )
        with patch(
            "deep_agent.aegra.mcp_resource_tools.read_resource",
            new_callable=AsyncMock,
        ) as mock_read:
            result = await _tool(tools, READ_TOOL).ainvoke(
                {"mcp_name": "s", "uri": "template://echo/a/b"}
            )
        mock_read.assert_not_awaited()
        assert "not allowed" in result
