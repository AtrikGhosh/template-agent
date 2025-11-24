"""Tests for the stream route."""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from template_agent.src.routes.stream import (
    message_generator,
    router,
    stream,
    _sse_response_example,
)
from template_agent.src.schema import StreamRequest


class TestMessageGenerator:
    """Test cases for the message_generator function."""

    @pytest.fixture
    def stream_request(self):
        """Create a sample stream request."""
        return StreamRequest(
            message="Hello, how can you help?",
            thread_id="thread_123",
            session_id="session_456",
            user_id="user_789",
            stream_tokens=True,
        )

    @pytest.mark.asyncio
    async def test_message_generator_success(self, stream_request):
        """Test successful message generation."""
        # Create mock agent manager
        mock_agent_manager = AsyncMock()

        async def mock_stream_response(request):
            yield {"type": "message", "content": {"type": "ai", "content": "Hello!"}}
            yield {"type": "token", "content": "How"}
            yield {"type": "token", "content": " are"}
            yield {"type": "token", "content": " you?"}

        mock_agent_manager.stream_response = mock_stream_response

        # Collect generated messages
        messages = []
        async for message in message_generator(stream_request, mock_agent_manager):
            messages.append(message)

        # Verify messages
        assert len(messages) == 5  # 4 events + [DONE]
        assert json.loads(messages[0])["type"] == "message"
        assert json.loads(messages[1])["type"] == "token"
        assert messages[-1] == "[DONE]\n\n"

    @pytest.mark.asyncio
    async def test_message_generator_filters_duplicate_human_messages(
        self, stream_request
    ):
        """Test that duplicate human messages are filtered out."""
        mock_agent_manager = AsyncMock()

        async def mock_stream_response(request):
            # First yield the duplicate human message
            yield {
                "type": "message",
                "content": {
                    "type": "human",
                    "content": "Hello, how can you help?",  # Same as input
                },
            }
            # Then yield AI response
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "I can help!"},
            }

        mock_agent_manager.stream_response = mock_stream_response

        messages = []
        async for message in message_generator(stream_request, mock_agent_manager):
            messages.append(message)

        # Should only have AI message and [DONE]
        parsed_messages = [json.loads(m) for m in messages[:-1]]  # Exclude [DONE]
        assert len(parsed_messages) == 1
        assert parsed_messages[0]["content"]["type"] == "ai"

    @pytest.mark.asyncio
    async def test_message_generator_handles_errors(self, stream_request):
        """Test error handling in message generator."""
        mock_agent_manager = AsyncMock()

        async def mock_stream_response(request):
            yield {
                "type": "message",
                "content": {"type": "ai", "content": "Starting..."},
            }
            raise Exception("Stream error occurred")

        mock_agent_manager.stream_response = mock_stream_response

        messages = []
        async for message in message_generator(stream_request, mock_agent_manager):
            messages.append(message)

        # Should have initial message, error event, and [DONE]
        assert len(messages) == 3
        error_event = json.loads(messages[1])
        assert error_event["type"] == "error"
        assert error_event["content"]["message"] == "Internal server error"
        assert error_event["content"]["recoverable"] is False
        assert error_event["content"]["error_type"] == "stream_error"
        assert messages[-1] == "[DONE]\n\n"

    @pytest.mark.asyncio
    async def test_message_generator_logging(self, stream_request):
        """Test that message generator logs appropriately."""
        mock_agent_manager = AsyncMock()

        async def mock_stream_response(request):
            yield {"type": "message", "content": {"type": "ai", "content": "Response"}}

        mock_agent_manager.stream_response = mock_stream_response

        with patch("template_agent.src.routes.stream.app_logger") as mock_logger:
            messages = []
            async for message in message_generator(stream_request, mock_agent_manager):
                messages.append(message)

            # Verify logging
            mock_logger.info.assert_called_once()
            assert "Starting stream for message" in mock_logger.info.call_args[0][0]

    @pytest.mark.asyncio
    async def test_message_generator_empty_stream(self, stream_request):
        """Test message generator with empty stream."""
        mock_agent_manager = AsyncMock()

        async def mock_stream_response(request):
            # Empty generator - yields nothing
            return
            yield  # Never reached

        mock_agent_manager.stream_response = mock_stream_response

        messages = []
        async for message in message_generator(stream_request, mock_agent_manager):
            messages.append(message)

        # Should only have [DONE]
        assert len(messages) == 1
        assert messages[0] == "[DONE]\n\n"


class TestSSEResponseExample:
    """Test cases for the SSE response example function."""

    def test_sse_response_example_structure(self):
        """Test that SSE response example has correct structure."""
        example = _sse_response_example()

        assert 200 in example
        response_200 = example[200]

        assert "description" in response_200
        assert "Simplified Format" in response_200["description"]

        assert "content" in response_200
        assert "text/event-stream" in response_200["content"]

        stream_content = response_200["content"]["text/event-stream"]
        assert "example" in stream_content
        assert "schema" in stream_content

    def test_sse_response_example_content(self):
        """Test that SSE response example contains valid JSON events."""
        example = _sse_response_example()
        example_text = example[200]["content"]["text/event-stream"]["example"]

        # Split into individual events
        events = example_text.strip().split("\n\n")

        # Should have multiple events
        assert len(events) > 1

        # Last event should be [DONE]
        assert events[-1] == "[DONE]"

        # Parse other events as JSON
        for event in events[:-1]:
            parsed = json.loads(event)
            assert "type" in parsed
            assert "content" in parsed


class TestStreamEndpoint:
    """Test cases for the stream endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock(spec=Request)
        request.headers = {"X-Token": "test_token"}
        return request

    @pytest.fixture
    def stream_request_data(self):
        """Create stream request data."""
        return {
            "message": "Test message",
            "thread_id": "thread_123",
            "session_id": "session_456",
            "user_id": "user_789",
            "stream_tokens": True,
        }

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.stream.AgentManager")
    @patch("template_agent.src.routes.stream.message_generator")
    async def test_stream_endpoint_success(
        self,
        mock_message_gen,
        mock_agent_manager_class,
        mock_request,
        stream_request_data,
    ):
        """Test successful stream endpoint execution."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_agent_manager_class.return_value = mock_manager

        async def mock_generator(request, manager):
            yield "event1\n\n"
            yield "event2\n\n"
            yield "[DONE]\n\n"

        mock_message_gen.side_effect = mock_generator

        # Create stream request
        stream_request = StreamRequest(**stream_request_data)

        # Call endpoint
        response = await stream(stream_request, mock_request)

        # Verify response
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Connection"] == "keep-alive"

        # Verify AgentManager was created with token
        mock_agent_manager_class.assert_called_once_with(redhat_sso_token="test_token")

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.stream.AgentManager")
    async def test_stream_endpoint_without_token(
        self, mock_agent_manager_class, stream_request_data
    ):
        """Test stream endpoint without authentication token."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}  # No X-Token

        mock_manager = MagicMock()
        mock_agent_manager_class.return_value = mock_manager

        stream_request = StreamRequest(**stream_request_data)

        with patch("template_agent.src.routes.stream.message_generator"):
            response = await stream(stream_request, mock_request)

        # Should still work but with no token
        mock_agent_manager_class.assert_called_once_with(redhat_sso_token=None)
        assert isinstance(response, StreamingResponse)

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.stream.AgentManager")
    async def test_stream_endpoint_initialization_error(
        self, mock_agent_manager_class, mock_request, stream_request_data
    ):
        """Test stream endpoint handles initialization errors."""
        # Make AgentManager initialization fail
        mock_agent_manager_class.side_effect = Exception("Initialization failed")

        stream_request = StreamRequest(**stream_request_data)

        with pytest.raises(HTTPException) as exc_info:
            await stream(stream_request, mock_request)

        assert exc_info.value.status_code == 500
        assert "Failed to initialize agent" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.stream.AgentManager")
    @patch("template_agent.src.routes.stream.app_logger")
    async def test_stream_endpoint_logging(
        self, mock_logger, mock_agent_manager_class, mock_request, stream_request_data
    ):
        """Test that stream endpoint logs appropriately."""
        mock_manager = MagicMock()
        mock_agent_manager_class.return_value = mock_manager

        stream_request = StreamRequest(**stream_request_data)

        with patch("template_agent.src.routes.stream.message_generator"):
            await stream(stream_request, mock_request)

        # Verify logging
        assert mock_logger.info.call_count == 1
        log_message = mock_logger.info.call_args[0][0]
        assert "Received token: Yes" in log_message


class TestStreamIntegration:
    """Integration tests for the stream route."""

    def test_stream_route_registered(self):
        """Test that stream route is registered correctly."""
        routes = [route.path for route in router.routes]
        assert "/v1/stream" in routes

    def test_stream_route_methods(self):
        """Test that stream route accepts POST method."""
        for route in router.routes:
            if route.path == "/v1/stream":
                assert "POST" in route.methods

    @pytest.mark.asyncio
    async def test_stream_with_test_client(self):
        """Test stream endpoint with FastAPI test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch(
            "template_agent.src.routes.stream.AgentManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()

            async def mock_stream_response(request):
                yield {"type": "message", "content": {"type": "ai", "content": "Test"}}

            mock_manager.stream_response = mock_stream_response
            mock_manager_class.return_value = mock_manager

            # Use TestClient for synchronous testing
            client = TestClient(app)

            request_data = {
                "message": "Test",
                "session_id": "session_123",
                "user_id": "user_456",
            }

            # Note: TestClient doesn't properly support SSE streaming,
            # but we can verify the endpoint is callable
            with patch(
                "template_agent.src.routes.stream.message_generator"
            ) as mock_gen:

                async def simple_gen(req, mgr):
                    yield '{"type": "message", "content": {"type": "ai", "content": "Test"}}\n\n'
                    yield "[DONE]\n\n"

                mock_gen.return_value = simple_gen(None, None)

                response = client.post(
                    "/v1/stream", json=request_data, headers={"X-Token": "test"}
                )

                # Should return 200 with streaming response
                assert response.status_code == 200

    def test_stream_request_validation(self):
        """Test that stream endpoint validates request properly."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        # Test with invalid request (missing required field)
        invalid_request = {"thread_id": "123"}  # Missing 'message'

        response = client.post("/v1/stream", json=invalid_request)

        # Should return validation error
        assert response.status_code == 422  # Unprocessable Entity

    def test_stream_response_headers(self):
        """Test that stream response has correct headers."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch(
            "template_agent.src.routes.stream.AgentManager"
        ) as mock_manager_class:
            with patch(
                "template_agent.src.routes.stream.message_generator"
            ) as mock_gen:
                mock_manager_class.return_value = MagicMock()

                async def simple_gen(req, mgr):
                    yield "[DONE]\n\n"

                mock_gen.return_value = simple_gen(None, None)

                request_data = {"message": "Test"}

                response = client.post("/v1/stream", json=request_data)

                # Check headers
                assert response.headers.get("cache-control") == "no-cache"
                assert response.headers.get("connection") == "keep-alive"


class TestStreamRequestHandling:
    """Test cases for StreamRequest handling."""

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.stream.AgentManager")
    @patch("template_agent.src.routes.stream.message_generator")
    async def test_stream_with_minimal_request(
        self, mock_message_gen, mock_agent_manager_class
    ):
        """Test stream with minimal request (only message field)."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}

        mock_manager = MagicMock()
        mock_agent_manager_class.return_value = mock_manager

        async def mock_generator(request, manager):
            yield "[DONE]\n\n"

        mock_message_gen.side_effect = mock_generator

        # Minimal request
        stream_request = StreamRequest(message="Hello")

        response = await stream(stream_request, mock_request)

        assert isinstance(response, StreamingResponse)
        # Verify defaults were used
        call_args = mock_message_gen.call_args[0][0]
        assert call_args.message == "Hello"
        assert call_args.stream_tokens is True  # Default value

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.stream.AgentManager")
    @patch("template_agent.src.routes.stream.message_generator")
    async def test_stream_with_complete_request(
        self, mock_message_gen, mock_agent_manager_class
    ):
        """Test stream with all request fields."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Token": "auth_token"}

        mock_manager = MagicMock()
        mock_agent_manager_class.return_value = mock_manager

        async def mock_generator(request, manager):
            yield "[DONE]\n\n"

        mock_message_gen.side_effect = mock_generator

        # Complete request
        stream_request = StreamRequest(
            message="Hello",
            thread_id="thread_123",
            session_id="session_456",
            user_id="user_789",
            stream_tokens=False,
        )

        response = await stream(stream_request, mock_request)

        assert isinstance(response, StreamingResponse)
        # Verify all fields were passed
        call_args = mock_message_gen.call_args[0][0]
        assert call_args.message == "Hello"
        assert call_args.thread_id == "thread_123"
        assert call_args.session_id == "session_456"
        assert call_args.user_id == "user_789"
        assert call_args.stream_tokens is False
