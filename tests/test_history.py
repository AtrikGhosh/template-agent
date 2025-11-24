"""Tests for the history route."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from template_agent.src.routes.history import history, router
from template_agent.src.schema import ChatHistoryResponse, ChatMessage


class TestHistoryEndpoint:
    """Test cases for the history endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock(spec=Request)
        request.headers = {"X-Token": "test_token"}
        return request

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.psycopg2")
    async def test_history_postgresql_success(
        self, mock_psycopg2, mock_settings, mock_request
    ):
        """Test successful history retrieval from PostgreSQL."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        # Mock database response
        checkpoint_data = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ]
            }
        }
        metadata = {
            "run_id": "run_123",
            "session_id": "session_456",
            "user_id": "user_789",
        }
        mock_cursor.fetchone.return_value = (checkpoint_data, metadata)

        # Call the endpoint
        response = await history("thread_123", mock_request)

        # Verify response
        assert isinstance(response, ChatHistoryResponse)
        assert len(response.messages) == 2
        assert response.messages[0].type == "human"
        assert response.messages[0].content == "Hello"
        assert response.messages[1].type == "ai"
        assert response.messages[1].content == "Hi there!"

        # Verify database query
        mock_cursor.execute.assert_called_once()
        query = mock_cursor.execute.call_args[0][0]
        assert "SELECT checkpoint, metadata FROM checkpoints" in query
        assert "thread_id = %s" in query

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.get_shared_checkpointer")
    async def test_history_inmemory_success(
        self, mock_get_checkpointer, mock_settings, mock_request
    ):
        """Test successful history retrieval from in-memory storage."""
        mock_settings.USE_INMEMORY_SAVER = True

        # Mock checkpointer
        mock_checkpointer = MagicMock()
        mock_get_checkpointer.return_value = mock_checkpointer

        # Mock checkpoint state
        mock_checkpoint_tuple = MagicMock()
        mock_checkpoint_tuple.checkpoint = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Question?"),
                    AIMessage(content="Answer!"),
                ]
            }
        }
        mock_checkpointer.list.return_value = [mock_checkpoint_tuple]

        # Call the endpoint
        response = await history("thread_123", mock_request)

        # Verify response
        assert isinstance(response, ChatHistoryResponse)
        assert len(response.messages) == 2
        assert response.messages[0].type == "human"
        assert response.messages[0].content == "Question?"
        assert response.messages[1].type == "ai"
        assert response.messages[1].content == "Answer!"

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.get_shared_checkpointer")
    async def test_history_inmemory_no_checkpoints(
        self, mock_get_checkpointer, mock_settings, mock_request
    ):
        """Test history with no checkpoints in memory."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_checkpointer = MagicMock()
        mock_get_checkpointer.return_value = mock_checkpointer
        mock_checkpointer.list.return_value = []  # No checkpoints

        response = await history("thread_123", mock_request)

        assert isinstance(response, ChatHistoryResponse)
        assert len(response.messages) == 0

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.get_shared_checkpointer")
    async def test_history_inmemory_error_handling(
        self, mock_get_checkpointer, mock_settings, mock_request
    ):
        """Test error handling for in-memory storage."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_checkpointer = MagicMock()
        mock_get_checkpointer.return_value = mock_checkpointer
        mock_checkpointer.list.side_effect = Exception("Memory error")

        response = await history("thread_123", mock_request)

        # Should return empty history on error
        assert isinstance(response, ChatHistoryResponse)
        assert len(response.messages) == 0

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.psycopg2")
    async def test_history_postgresql_no_data(
        self, mock_psycopg2, mock_settings, mock_request
    ):
        """Test history with no data in PostgreSQL."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        # No data found
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []

        response = await history("thread_123", mock_request)

        assert isinstance(response, ChatHistoryResponse)
        assert len(response.messages) == 0

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.psycopg2")
    async def test_history_postgresql_connection_error(
        self, mock_psycopg2, mock_settings, mock_request
    ):
        """Test PostgreSQL connection error handling."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_psycopg2.connect.side_effect = Exception("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await history("thread_123", mock_request)

        assert exc_info.value.status_code == 500
        assert "Failed to retrieve chat history" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.psycopg2")
    async def test_history_postgresql_fallback_processing(
        self, mock_psycopg2, mock_settings, mock_request
    ):
        """Test fallback processing when latest checkpoint has no messages."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        # Latest checkpoint has no messages in channel_values
        mock_cursor.fetchone.return_value = ({"channel_values": {}}, {})

        # Fallback query returns checkpoints with writes
        checkpoint_with_writes = (
            {},  # checkpoint_data
            {  # metadata
                "writes": {
                    "__start__": {
                        "messages": [
                            {
                                "kwargs": {
                                    "type": "human",
                                    "content": "Fallback message",
                                }
                            }
                        ]
                    }
                }
            },
        )
        mock_cursor.fetchall.return_value = [checkpoint_with_writes]

        response = await history("thread_123", mock_request)

        # Should process fallback messages
        assert isinstance(response, ChatHistoryResponse)
        assert len(response.messages) == 1
        assert response.messages[0].type == "human"
        assert response.messages[0].content == "Fallback message"

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.logger")
    async def test_history_logging(self, mock_logger, mock_request):
        """Test that history endpoint logs appropriately."""
        with patch("template_agent.src.routes.history.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = True
            with patch(
                "template_agent.src.routes.history.get_shared_checkpointer"
            ) as mock_get:
                mock_checkpointer = MagicMock()
                mock_get.return_value = mock_checkpointer
                mock_checkpointer.list.return_value = []

                await history("thread_123", mock_request)

                # Verify logging
                assert mock_logger.info.call_count >= 2
                log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any(
                    "Retrieving history for thread_id: thread_123" in msg
                    for msg in log_messages
                )
                assert any("Access token present: True" in msg for msg in log_messages)


class TestHistoryMessageConversion:
    """Test cases for message conversion in history."""

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.get_shared_checkpointer")
    async def test_history_handles_tool_messages(
        self, mock_get_checkpointer, mock_settings
    ):
        """Test that tool messages are converted correctly."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        mock_checkpointer = MagicMock()
        mock_get_checkpointer.return_value = mock_checkpointer

        # Include a tool message
        mock_checkpoint_tuple = MagicMock()
        mock_checkpoint_tuple.checkpoint = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Calculate 2+2"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "calculator",
                                "args": {"a": 2, "b": 2},
                                "id": "call_123",
                            }
                        ],
                    ),
                    ToolMessage(content="4", tool_call_id="call_123"),
                    AIMessage(content="The answer is 4"),
                ]
            }
        }
        mock_checkpointer.list.return_value = [mock_checkpoint_tuple]

        response = await history("thread_123", mock_request)

        assert len(response.messages) == 4
        assert response.messages[2].type == "tool"
        assert response.messages[2].content == "4"
        assert response.messages[2].tool_call_id == "call_123"

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.history.settings")
    @patch("template_agent.src.routes.history.get_shared_checkpointer")
    async def test_history_deduplicates_messages(
        self, mock_get_checkpointer, mock_settings
    ):
        """Test that duplicate messages are deduplicated when processing all checkpoints."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        mock_checkpointer = MagicMock()
        mock_get_checkpointer.return_value = mock_checkpointer

        # Multiple checkpoints with overlapping messages
        checkpoint1 = MagicMock()
        checkpoint1.checkpoint = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Hello"),
                ]
            }
        }

        checkpoint2 = MagicMock()
        checkpoint2.checkpoint = {
            "channel_values": {
                "messages": [
                    HumanMessage(content="Hello"),  # Duplicate
                    AIMessage(content="Hi"),
                ]
            }
        }

        mock_checkpointer.list.return_value = [checkpoint1, checkpoint2]

        with patch(
            "template_agent.src.routes.history.langchain_to_chat_message"
        ) as mock_convert:
            # Mock conversion to return messages with empty content first time
            msg1 = ChatMessage(type="human", content="")
            msg2 = ChatMessage(type="ai", content="Hi")
            mock_convert.side_effect = [
                msg1,
                msg2,
            ]  # Only convert from latest checkpoint

            response = await history("thread_123", mock_request)

            # Latest checkpoint approach should work
            assert len(response.messages) == 2


class TestHistoryIntegration:
    """Integration tests for the history route."""

    def test_history_route_registered(self):
        """Test that history route is registered correctly."""
        routes = [route.path for route in router.routes]
        assert "/v1/history/{thread_id}" in routes

    def test_history_route_methods(self):
        """Test that history route accepts GET method."""
        for route in router.routes:
            if "/v1/history/" in route.path:
                assert "GET" in route.methods

    def test_history_with_test_client(self):
        """Test history endpoint with FastAPI test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("template_agent.src.routes.history.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = True
            with patch(
                "template_agent.src.routes.history.get_shared_checkpointer"
            ) as mock_get:
                mock_checkpointer = MagicMock()
                mock_get.return_value = mock_checkpointer
                mock_checkpointer.list.return_value = []

                client = TestClient(app)
                response = client.get(
                    "/v1/history/thread_123", headers={"X-Token": "test"}
                )

                assert response.status_code == 200
                data = response.json()
                assert "messages" in data
                assert isinstance(data["messages"], list)
