"""Unit tests for MCP Apps host proxy (resources/read + tools/call)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from mcp import types

from deep_agent.aegra.mcp_host import (
    call_app_tool,
    list_resource_templates,
    list_resources,
    read_resource,
)


def _server_cfg(**overrides):
    cfg = {
        "url": "http://localhost:5003/mcp",
        "transport": "streamable_http",
        "enabled": True,
        "auth": False,
        "auth_mode": "sso",
        "ssl_verify": False,
        "timeout": 30,
    }
    cfg.update(overrides)
    return cfg


@asynccontextmanager
async def _fake_session(session):
    yield session


class TestReadResource:
    @pytest.mark.asyncio
    async def test_rejects_empty_uri(self):
        with pytest.raises(HTTPException) as exc:
            await read_resource(
                "charts",
                "",
                user_id="u1",
                sso_token=None,
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_server_404(self):
        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={},
            ),
            pytest.raises(HTTPException) as exc,
        ):
            await read_resource(
                "missing",
                "ui://charts/app.html",
                user_id="u1",
                sso_token=None,
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_reads_any_resource_uri(self):
        content = types.TextResourceContents.model_validate(
            {
                "uri": "showcase://sample.json",
                "mimeType": "application/json",
                "text": '{"ok": true}',
            }
        )
        session = MagicMock()
        session.read_resource = AsyncMock(
            return_value=types.ReadResourceResult(contents=[content])
        )

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            result = await read_resource(
                "charts",
                "showcase://sample.json",
                user_id="u1",
                sso_token=None,
            )

        session.read_resource.assert_awaited_once()
        assert result["contents"][0]["text"] == '{"ok": true}'


class TestListResources:
    @pytest.mark.asyncio
    async def test_lists_resources(self):
        session = MagicMock()
        session.list_resources = AsyncMock(
            return_value=types.ListResourcesResult(
                resources=[
                    types.Resource.model_validate(
                        {
                            "uri": "showcase://sample.json",
                            "name": "sample",
                            "mimeType": "application/json",
                        }
                    )
                ]
            )
        )

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            result = await list_resources(
                "charts",
                user_id="u1",
                sso_token=None,
            )

        session.list_resources.assert_awaited_once_with(cursor=None)
        assert result["resources"][0]["uri"] == "showcase://sample.json"


class TestListResourceTemplates:
    @pytest.mark.asyncio
    async def test_lists_resource_templates(self):
        session = MagicMock()
        session.list_resource_templates = AsyncMock(
            return_value=types.ListResourceTemplatesResult(
                resourceTemplates=[
                    types.ResourceTemplate.model_validate(
                        {
                            "uriTemplate": "showcase://{id}",
                            "name": "sample_template",
                        }
                    )
                ]
            )
        )

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            result = await list_resource_templates(
                "charts",
                user_id="u1",
                sso_token=None,
            )

        session.list_resource_templates.assert_awaited_once_with(cursor=None)
        templates = result.get("resourceTemplates") or result.get("resource_templates")
        assert templates[0]["name"] == "sample_template"


class TestCallAppTool:
    @pytest.mark.asyncio
    async def test_rejects_model_only_tool(self):
        tool = types.Tool.model_validate(
            {
                "name": "secret_admin",
                "inputSchema": {"type": "object"},
                "_meta": {"ui": {"visibility": ["model"]}},
            }
        )
        session = MagicMock()
        session.list_tools = AsyncMock(return_value=types.ListToolsResult(tools=[tool]))
        session.call_tool = AsyncMock()

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
            pytest.raises(HTTPException) as exc,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            await call_app_tool(
                "charts",
                "secret_admin",
                {},
                user_id="u1",
                sso_token=None,
            )

        assert exc.value.status_code == 403
        assert exc.value.detail["error"] == "tool_not_app_callable"
        assert exc.value.detail["tool"] == "secret_admin"
        assert exc.value.detail["visibility"] == ["model"]
        session.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_app_visible_tool(self):
        tool = types.Tool.model_validate(
            {
                "name": "refresh_showcase",
                "inputSchema": {"type": "object"},
                "_meta": {
                    "ui": {"visibility": ["app"], "resourceUri": "ui://x"},
                },
            }
        )
        session = MagicMock()
        session.list_tools = AsyncMock(return_value=types.ListToolsResult(tools=[tool]))
        session.call_tool = AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                structuredContent={"status": "refreshed"},
            )
        )

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            result = await call_app_tool(
                "charts",
                "refresh_showcase",
                {"topic": "demo"},
                user_id="u1",
                sso_token=None,
            )

        session.call_tool.assert_awaited_once_with(
            "refresh_showcase", {"topic": "demo"}
        )
        assert result["structuredContent"]["status"] == "refreshed"

    @pytest.mark.asyncio
    async def test_default_visibility_allows_app_call(self):
        """Tools without visibility default to model+app and remain callable."""
        tool = types.Tool.model_validate(
            {
                "name": "show_chart",
                "inputSchema": {"type": "object"},
                "_meta": {"ui": {"resourceUri": "ui://charts/app.html"}},
            }
        )
        session = MagicMock()
        session.list_tools = AsyncMock(return_value=types.ListToolsResult(tools=[tool]))
        session.call_tool = AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
            )
        )

        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={"charts": _server_cfg()},
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_host.MultiServerMCPClient",
            ) as mock_client_cls,
        ):
            client = MagicMock()
            client.session = lambda _name: _fake_session(session)
            mock_client_cls.return_value = client

            result = await call_app_tool(
                "charts",
                "show_chart",
                None,
                user_id="u1",
                sso_token=None,
            )

        session.call_tool.assert_awaited_once_with("show_chart", {})
        assert result["content"][0]["text"] == "ok"

    @pytest.mark.asyncio
    async def test_oauth_missing_token_returns_401(self):
        with (
            patch(
                "deep_agent.aegra.mcp_host._get_server_configs",
                return_value={
                    "preset": _server_cfg(auth=True, auth_mode="oauth"),
                },
            ),
            patch(
                "deep_agent.aegra.mcp_host._resolve_connection_token",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_agent.aegra.mcp_auth.get_mcp_credential_resolver",
            ) as mock_resolver,
            pytest.raises(HTTPException) as exc,
        ):
            mock_resolver.return_value.connect_url.return_value = "/mcp/preset/connect"
            await call_app_tool(
                "preset",
                "refresh",
                {},
                user_id="u1",
                sso_token=None,
            )

        assert exc.value.status_code == 401
        assert exc.value.detail["error"] == "authorization_required"
        assert exc.value.detail["connect_url"] == "/mcp/preset/connect"


class TestRouteWiring:
    """Smoke-test FastAPI route handlers parse bodies and delegate."""

    @pytest.mark.asyncio
    async def test_resources_read_route(self):
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(return_value={"uri": "ui://x"})
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.read_resource",
                new_callable=AsyncMock,
                return_value={"contents": []},
            ) as mock_read,
        ):
            response = await mcp_routes.mcp_resources_read("charts", request)

        assert response.status_code == 200
        mock_read.assert_awaited_once_with(
            "charts",
            "ui://x",
            user_id="user-1",
            sso_token="tok",
        )

    @pytest.mark.asyncio
    async def test_resources_list_route(self):
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(return_value={})
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.list_resources",
                new_callable=AsyncMock,
                return_value={"resources": []},
            ) as mock_list,
        ):
            response = await mcp_routes.mcp_resources_list("charts", request)

        assert response.status_code == 200
        mock_list.assert_awaited_once_with(
            "charts",
            cursor=None,
            user_id="user-1",
            sso_token="tok",
        )

    @pytest.mark.asyncio
    async def test_tools_list_route(self):
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(return_value={})
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.list_tools",
                new_callable=AsyncMock,
                return_value={
                    "tools": [
                        {
                            "name": "show_chart",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            ) as mock_list,
        ):
            response = await mcp_routes.mcp_tools_list("charts", request)

        assert response.status_code == 200
        mock_list.assert_awaited_once_with(
            "charts",
            cursor=None,
            user_id="user-1",
            sso_token="tok",
        )

    @pytest.mark.asyncio
    async def test_tools_call_route_returns_dict_http_errors_as_json(self):
        """Dict HTTPException.detail must not hit AgentProtocolError (string message)."""
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"name": "hostile_model_only", "arguments": {}}
        )
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.call_app_tool",
                new_callable=AsyncMock,
                side_effect=HTTPException(
                    status_code=403,
                    detail={
                        "error": "tool_not_app_callable",
                        "tool": "hostile_model_only",
                        "visibility": ["model"],
                    },
                ),
            ),
        ):
            response = await mcp_routes.mcp_tools_call("mcp-app-test", request)

        assert response.status_code == 403
        assert response.body is not None
        import json

        body = json.loads(response.body)
        assert body["error"] == "tool_not_app_callable"
        assert body["tool"] == "hostile_model_only"

    @pytest.mark.asyncio
    async def test_resources_read_route_returns_dict_http_errors_as_json(self):
        """authorization_required dict detail must return 401 JSON, not 500."""
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(return_value={"uri": "ui://charts/app.html"})
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.read_resource",
                new_callable=AsyncMock,
                side_effect=HTTPException(
                    status_code=401,
                    detail={
                        "error": "authorization_required",
                        "mcp_name": "charts",
                        "connect_url": "/mcp/charts/connect",
                    },
                ),
            ),
        ):
            response = await mcp_routes.mcp_resources_read("charts", request)

        assert response.status_code == 401
        import json

        body = json.loads(response.body)
        assert body["error"] == "authorization_required"
        assert body["mcp_name"] == "charts"

    @pytest.mark.asyncio
    async def test_resources_list_route_returns_dict_http_errors_as_json(self):
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(return_value={})
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.list_resources",
                new_callable=AsyncMock,
                side_effect=HTTPException(
                    status_code=401,
                    detail={
                        "error": "authorization_required",
                        "mcp_name": "charts",
                        "connect_url": "/mcp/charts/connect",
                    },
                ),
            ),
        ):
            response = await mcp_routes.mcp_resources_list("charts", request)

        assert response.status_code == 401
        import json

        assert json.loads(response.body)["error"] == "authorization_required"

    @pytest.mark.asyncio
    async def test_tools_call_route(self):
        from deep_agent.aegra import mcp_routes

        request = MagicMock()
        request.json = AsyncMock(
            return_value={"name": "refresh_showcase", "arguments": {"a": 1}}
        )
        request.headers = {"authorization": "Bearer tok"}

        with (
            patch(
                "deep_agent.aegra.mcp_routes._authenticated_user_id",
                new_callable=AsyncMock,
                return_value="user-1",
            ),
            patch(
                "deep_agent.aegra.mcp_host.call_app_tool",
                new_callable=AsyncMock,
                return_value={"content": []},
            ) as mock_call,
        ):
            response = await mcp_routes.mcp_tools_call("charts", request)

        assert response.status_code == 200
        mock_call.assert_awaited_once_with(
            "charts",
            "refresh_showcase",
            {"a": 1},
            user_id="user-1",
            sso_token="tok",
        )
