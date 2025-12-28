"""Tests for the AgentManager class."""

import json
from collections.abc import AsyncGenerator
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command, Interrupt

from template_agent.src.core.manager import AgentManager
from template_agent.src.schema import StreamRequest


class TestAgentManager:
    """Test cases for the AgentManager class."""

    @pytest.fixture
    def stream_request(self):
        """Create a sample stream request."""
        return StreamRequest(
            message="Hello, how are you?",
            thread_id="thread_123",
            session_id="session_456",
            user_id="user_789",
            stream_tokens=True,
        )

    @pytest.fixture
    def agent_manager(self):
        """Create an AgentManager instance."""
        return AgentManager(redhat_sso_token="test_token")

    @pytest.mark.asyncio
    async def test_agent_manager_initialization(self):
        """Test AgentManager initialization."""
        # Test with token
        manager = AgentManager(redhat_sso_token="test_token")
        assert manager.redhat_sso_token == "test_token"
        assert manager._agent is None
        assert manager._current_tool_call_id is None

        # Test without token
        manager = AgentManager()
        assert manager.redhat_sso_token is None

    @pytest.mark.asyncio
    @patch("template_agent.src.core.manager.get_template_agent")
    @patch("template_agent.src.core.manager.register_thread")
    async def test_stream_response_success(
        self, mock_register_thread, mock_get_agent, agent_manager, stream_request
    ):
        """Test successful stream_response execution."""
        # Create mock agent
        mock_agent = AsyncMock()
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        # Mock agent.aget_state
        mock_state = MagicMock()
        mock_state.tasks = []
        mock_agent.aget_state.return_value = mock_state

        # Mock agent.astream
        async def mock_astream(*args, **kwargs):
            # Yield update events
            yield ("updates", {"agent": {"messages": [AIMessage(content="Hello!")]}})
            # Yield token events
            yield ("messages", (AIMessageChunk(content="How"), {}))
            yield ("messages", (AIMessageChunk(content=" are"), {}))
            yield ("messages", (AIMessageChunk(content=" you?"), {}))

        mock_agent.astream = mock_astream
        mock_get_agent.return_value = mock_agent

        # Execute stream_response
        events = []
        async for event in agent_manager.stream_response(stream_request):
            events.append(event)

        # Verify events were generated
        assert len(events) > 0
        # Check for message event
        message_events = [e for e in events if e.get("type") == "message"]
        assert len(message_events) > 0
        # Check for token events
        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) > 0

        # Verify get_template_agent was called correctly
        mock_get_agent.assert_called_once_with(
            "test_token",
            enable_checkpointing=True,
            user_id="user_789",
            message="Hello, how are you?",
        )

    @pytest.mark.asyncio
    @patch("template_agent.src.core.manager.get_template_agent")
    async def test_stream_response_handles_errors(
        self, mock_get_agent, agent_manager, stream_request
    ):
        """Test stream_response error handling."""
        # Create mock agent that raises an error
        mock_agent = AsyncMock()
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)
        mock_agent.aget_state.side_effect = Exception("Test error")
        mock_get_agent.return_value = mock_agent

        # Execute stream_response
        events = []
        async for event in agent_manager.stream_response(stream_request):
            events.append(event)

        # Should yield error event
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["content"]["message"] == "Internal server error"
        assert events[0]["content"]["recoverable"] is False

    @pytest.mark.asyncio
    @patch("template_agent.src.core.manager.register_thread")
    @patch("template_agent.src.core.manager.settings")
    async def test_handle_input_with_thread_id(
        self, mock_settings, mock_register_thread, agent_manager, stream_request
    ):
        """Test _handle_input with provided thread_id."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_agent = AsyncMock()
        mock_state = MagicMock()
        mock_state.tasks = []
        mock_agent.aget_state.return_value = mock_state

        kwargs, run_id, thread_id = await agent_manager._handle_input(
            stream_request, mock_agent
        )

        # Verify thread_id is used
        assert thread_id == "thread_123"
        assert kwargs["config"]["configurable"]["thread_id"] == "thread_123"
        assert kwargs["input"]["messages"][0].content == "Hello, how are you?"

        # Verify thread registration
        mock_register_thread.assert_called_once_with("user_789", "thread_123")

    @pytest.mark.asyncio
    @patch("template_agent.src.core.manager.uuid4")
    @patch("template_agent.src.core.manager.settings")
    async def test_handle_input_without_thread_id(
        self, mock_settings, mock_uuid4, agent_manager
    ):
        """Test _handle_input generates thread_id when not provided."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_uuid = "generated_thread_id"
        mock_uuid4.return_value = mock_uuid

        request = StreamRequest(
            message="Hello", thread_id=None, user_id="user_123", stream_tokens=True
        )

        mock_agent = AsyncMock()
        mock_state = MagicMock()
        mock_state.tasks = []
        mock_agent.aget_state.return_value = mock_state

        kwargs, run_id, thread_id = await agent_manager._handle_input(
            request, mock_agent
        )

        # Verify generated thread_id is used
        assert thread_id == mock_uuid
        assert kwargs["config"]["configurable"]["thread_id"] == mock_uuid

    @pytest.mark.asyncio
    async def test_handle_input_with_interrupt(self, agent_manager, stream_request):
        """Test _handle_input handles interrupts correctly."""
        mock_agent = AsyncMock()

        # Create mock state with interrupted task
        mock_state = MagicMock()
        mock_task = MagicMock()
        mock_task.interrupts = [Interrupt(value="Interrupted task")]
        mock_state.tasks = [mock_task]
        mock_agent.aget_state.return_value = mock_state

        kwargs, run_id, thread_id = await agent_manager._handle_input(
            stream_request, mock_agent
        )

        # Verify Command is used for resuming
        assert isinstance(kwargs["input"], Command)
        assert kwargs["input"].resume == "Hello, how are you?"

    def test_format_events_message_type(self, agent_manager):
        """Test _format_events for message events."""
        event = {"agent": {"messages": [AIMessage(content="Test message")]}}

        events = agent_manager._format_events(
            "updates", event, False, "run_123", "thread_123", "session_456"
        )

        assert len(events) == 1
        assert events[0]["type"] == "message"
        assert events[0]["content"]["type"] == "ai"
        assert events[0]["content"]["content"] == "Test message"
        assert events[0]["content"]["thread_id"] == "thread_123"
        assert events[0]["content"]["session_id"] == "session_456"

    def test_format_events_token_type(self, agent_manager):
        """Test _format_events for token events."""
        msg = AIMessageChunk(content="Hello")
        event = (msg, {"tags": []})

        events = agent_manager._format_events(
            "messages", event, True, "run_123", "thread_123", "session_456"
        )

        assert len(events) == 1
        assert events[0]["type"] == "token"
        assert events[0]["content"] == "Hello"

    def test_format_events_skips_non_stream_tokens(self, agent_manager):
        """Test _format_events skips tokens with skip_stream tag."""
        msg = AIMessageChunk(content="Hello")
        event = (msg, {"tags": ["skip_stream"]})

        events = agent_manager._format_events(
            "messages", event, True, "run_123", "thread_123", "session_456"
        )

        assert len(events) == 0

    def test_format_events_custom_type(self, agent_manager):
        """Test _format_events for custom events."""
        event = HumanMessage(content="Custom event")

        events = agent_manager._format_events(
            "custom", event, False, "run_123", "thread_123", "session_456"
        )

        assert len(events) == 1
        assert events[0]["type"] == "message"
        assert events[0]["content"]["type"] == "human"
        assert events[0]["content"]["content"] == "Custom event"

    def test_handle_update_events_with_interrupt(self, agent_manager):
        """Test _handle_update_events with interrupt."""
        interrupt = Interrupt(value="Please provide more information")
        event = {"__interrupt__": [interrupt]}

        events = agent_manager._handle_update_events(
            event, "run_123", "thread_123", "session_456"
        )

        assert len(events) == 1
        assert events[0]["type"] == "message"
        assert events[0]["content"]["type"] == "ai"
        assert events[0]["content"]["content"] == "Please provide more information"

    def test_handle_update_events_supervisor(self, agent_manager):
        """Test _handle_update_events with supervisor node."""
        event = {
            "supervisor": {
                "messages": [
                    AIMessage(content="First"),
                    AIMessage(content="Second"),
                    AIMessage(content="Last"),
                ]
            }
        }

        events = agent_manager._handle_update_events(
            event, "run_123", "thread_123", "session_456"
        )

        # Should only include the last AI message from supervisor
        assert len(events) == 1
        assert events[0]["content"]["content"] == "Last"

    def test_handle_update_events_expert_nodes(self, agent_manager):
        """Test _handle_update_events with expert nodes."""
        for node in ["research_expert", "math_expert"]:
            event = {node: {"messages": [AIMessage(content=f"Result from {node}")]}}

            events = agent_manager._handle_update_events(
                event, "run_123", "thread_123", "session_456"
            )

            # Should convert to ToolMessage
            assert len(events) == 1
            assert events[0]["content"]["type"] == "tool"
            assert events[0]["content"]["content"] == f"Result from {node}"

    def test_handle_token_events_with_content(self, agent_manager):
        """Test _handle_token_events with valid content."""
        msg = AIMessageChunk(content="Test token")
        event = (msg, {"tags": []})

        result = agent_manager._handle_token_events(event)

        assert result is not None
        assert result["type"] == "token"
        assert result["content"] == "Test token"

    def test_handle_token_events_filters_non_ai(self, agent_manager):
        """Test _handle_token_events filters non-AIMessageChunk."""
        msg = HumanMessage(content="Human message")
        event = (msg, {"tags": []})

        result = agent_manager._handle_token_events(event)

        assert result is None

    def test_handle_custom_events_success(self, agent_manager):
        """Test _handle_custom_events with valid message."""
        event = AIMessage(content="Custom AI message")

        result = agent_manager._handle_custom_events(
            event, "run_123", "thread_123", "session_456"
        )

        assert result is not None
        assert result["type"] == "message"
        assert result["content"]["type"] == "ai"
        assert result["content"]["content"] == "Custom AI message"

    def test_handle_custom_events_error(self, agent_manager):
        """Test _handle_custom_events handles errors."""
        event = "Invalid event"

        with patch("template_agent.src.core.manager.app_logger") as mock_logger:
            result = agent_manager._handle_custom_events(
                event, "run_123", "thread_123", "session_456"
            )

            assert result is None
            mock_logger.error.assert_called_once()

    def test_process_message_tuples(self, agent_manager):
        """Test _process_message_tuples processing."""
        messages = [
            ("content", "Hello"),
            ("tool_calls", []),
            HumanMessage(content="User message"),
            ("content", "Another message"),
        ]

        result = agent_manager._process_message_tuples(messages)

        # Should have 3 messages: 2 AIMessages and 1 HumanMessage
        assert len(result) == 3
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hello"
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)

    def test_create_ai_message(self, agent_manager):
        """Test _create_ai_message creates valid AIMessage."""
        parts = {
            "content": "Test content",
            "tool_calls": [],
            "additional_kwargs": {},
            "invalid_field": "should be ignored",
        }

        result = agent_manager._create_ai_message(parts)

        assert isinstance(result, AIMessage)
        assert result.content == "Test content"
        assert result.tool_calls == []

    def test_convert_chat_message_to_simple_format(self, agent_manager):
        """Test _convert_chat_message_to_simple_format."""
        from template_agent.src.schema import ChatMessage

        chat_message = ChatMessage(
            type="ai",
            content="Test message",
            tool_calls=[{"name": "tool", "args": {}, "id": "123"}],
            run_id="run_123",
            response_metadata={"model": "test"},
            custom_data={"key": "value"},
        )

        result = agent_manager._convert_chat_message_to_simple_format(
            chat_message, "thread_123", "session_456"
        )

        assert result["type"] == "ai"
        assert result["content"] == "Test message"
        assert result["tool_calls"] == [{"name": "tool", "args": {}, "id": "123"}]
        assert result["run_id"] == "run_123"
        assert result["thread_id"] == "thread_123"
        assert result["session_id"] == "session_456"
        assert result["response_metadata"] == {"model": "test"}
        assert result["custom_data"] == {"key": "value"}

    def test_extract_tool_call_id_from_message(self, agent_manager):
        """Test _extract_tool_call_id_from_message."""
        # Test with tool_calls
        msg = AIMessageChunk(
            content="", tool_calls=[{"id": "call_123", "name": "test_tool", "args": {}}]
        )
        result = agent_manager._extract_tool_call_id_from_message(msg)
        assert result == "call_123"

        # Test with tool_call_chunks
        msg = AIMessageChunk(content="")
        msg.tool_call_chunks = [{"id": "chunk_456"}]
        result = agent_manager._extract_tool_call_id_from_message(msg)
        assert result == "chunk_456"

        # Test with no tool calls
        msg = AIMessageChunk(content="Plain message")
        result = agent_manager._extract_tool_call_id_from_message(msg)
        assert result is None

    def test_update_tool_call_tracking_updates(self, agent_manager):
        """Test _update_tool_call_tracking with update events."""
        # Test with tool calls in message
        event = {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "call_789", "name": "test_tool", "args": {}}
                        ],
                    )
                ]
            }
        }
        agent_manager._update_tool_call_tracking("updates", event)
        assert agent_manager._current_tool_call_id == "call_789"

        # Test with tool response
        event = {
            "tools": {
                "messages": [ToolMessage(content="Result", tool_call_id="response_123")]
            }
        }
        agent_manager._update_tool_call_tracking("updates", event)
        assert agent_manager._current_tool_call_id == "response_123"

    def test_update_tool_call_tracking_messages(self, agent_manager):
        """Test _update_tool_call_tracking with message events."""
        msg = AIMessageChunk(
            content="",
            tool_calls=[{"id": "msg_call_456", "name": "test_tool", "args": {}}],
        )
        event = (msg, {})

        agent_manager._update_tool_call_tracking("messages", event)
        assert agent_manager._current_tool_call_id == "msg_call_456"

    @pytest.mark.asyncio
    async def test_prepare_streaming_input_with_history(self, agent_manager):
        """Test _prepare_streaming_input_with_history."""
        request = StreamRequest(
            message="New message",
            thread_id="thread_123",
            session_id="session_456",
            user_id="user_789",
        )

        existing_state = MagicMock()
        existing_state.values = {
            "messages": [
                HumanMessage(content="Old message 1"),
                AIMessage(content="Old response 1"),
            ]
        }

        result = await agent_manager._prepare_streaming_input_with_history(
            request, existing_state, "run_123", "thread_123"
        )

        # Should include history plus new message
        assert len(result["input"]["messages"]) == 3
        assert result["input"]["messages"][0].content == "Old message 1"
        assert result["input"]["messages"][1].content == "Old response 1"
        assert result["input"]["messages"][2].content == "New message"
        assert result["config"]["configurable"]["thread_id"] == "thread_123"

    @pytest.mark.asyncio
    async def test_save_final_conversation_state_success(self, agent_manager):
        """Test _save_final_conversation_state successful save."""
        mock_agent = AsyncMock()
        mock_agent.aupdate_state = AsyncMock()

        mock_config = MagicMock()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        with patch("template_agent.src.core.manager.app_logger") as mock_logger:
            await agent_manager._save_final_conversation_state(
                mock_agent, mock_config, messages, "thread_123"
            )

            mock_agent.aupdate_state.assert_called_once_with(
                config=mock_config, values={"messages": messages}
            )
            # Verify success was logged
            assert any(
                "Successfully saved" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_save_final_conversation_state_handles_errors(self, agent_manager):
        """Test _save_final_conversation_state handles errors gracefully."""
        mock_agent = AsyncMock()
        mock_agent.aupdate_state.side_effect = Exception("Save failed")

        mock_config = MagicMock()
        messages = []

        with patch("template_agent.src.core.manager.app_logger") as mock_logger:
            # Should not raise exception
            await agent_manager._save_final_conversation_state(
                mock_agent, mock_config, messages, "thread_123"
            )

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert (
                "Error saving final conversation state"
                in mock_logger.error.call_args[0][0]
            )

    def test_handle_update_events_error(self, agent_manager):
        """Test _handle_update_events error handling (lines 368-370)."""
        from langchain_core.messages import AIMessage

        # Create an update event with a message
        event = {"agent": {"messages": [AIMessage(content="test message")]}}

        with patch(
            "template_agent.src.core.manager.langchain_to_chat_message"
        ) as mock_langchain_to_chat:
            # Mock langchain_to_chat_message to return a ChatMessage
            from template_agent.src.schema import ChatMessage

            mock_chat = ChatMessage(role="assistant", content="test", type="ai")
            mock_langchain_to_chat.return_value = mock_chat

            with patch.object(
                agent_manager, "_convert_chat_message_to_simple_format"
            ) as mock_convert:
                # Make _convert_chat_message_to_simple_format raise an exception
                mock_convert.side_effect = Exception("Formatting error")

                with patch("template_agent.src.core.manager.app_logger") as mock_logger:
                    result = agent_manager._handle_update_events(
                        event, "run_123", "thread_123", None
                    )

                    # Should return error event
                    assert len(result) == 1
                    assert result[0]["type"] == "error"
                    assert result[0]["content"]["message"] == "Message formatting error"
                    assert result[0]["content"]["recoverable"] is True
                    mock_logger.error.assert_called_with(
                        "Error formatting message: Formatting error"
                    )

    def test_handle_token_events_with_tool_id(self, agent_manager):
        """Test _handle_token_events with tool_call_id (lines 405, 408)."""
        from langchain_core.messages import AIMessageChunk

        # Test with tool_call_id from current tracking
        agent_manager._current_tool_call_id = "tool_123"

        msg = AIMessageChunk(content="test content", id="msg_1")
        metadata = {"tags": []}  # No skip_stream tag
        event = (msg, metadata)

        result = agent_manager._handle_token_events(event)

        # Should return token event with tool_call_id
        assert result is not None
        assert result["type"] == "token"
        assert result["content"] == "test content"
        assert result["tool_call_id"] == "tool_123"

        # Test with empty content (should return None) - line 408
        msg_empty = AIMessageChunk(content="", id="msg_2")
        event_empty = (msg_empty, metadata)
        result = agent_manager._handle_token_events(event_empty)
        assert result is None

    def test_convert_message_with_optional_fields(self, agent_manager):
        """Test _convert_chat_message_to_simple_format with tool_call_id and ai_call_id (lines 470, 478)."""
        from template_agent.src.schema import ChatMessage

        # Create a message with optional fields
        msg = ChatMessage(
            role="assistant",
            content="test",
            type="ai",
            tool_call_id="tool_call_123",  # Line 470
            ai_call_id="ai_call_123",  # Line 478
        )

        result = agent_manager._convert_chat_message_to_simple_format(
            msg, "thread_1", "session_1"
        )

        # Verify optional fields are included
        assert result["tool_call_id"] == "tool_call_123"
        assert result["ai_call_id"] == "ai_call_123"
        assert result["thread_id"] == "thread_1"
        assert result["session_id"] == "session_1"

    def test_extract_tool_call_id_from_tool_message(self, agent_manager):
        """Test _extract_tool_call_id_from_message with tool_call_id (lines 508, 511-513)."""
        from langchain_core.messages import AIMessageChunk

        # Test successful path with tool_call_id (line 508)
        # Create a message that doesn't have tool_calls or tool_call_chunks but has tool_call_id
        msg = MagicMock(spec=AIMessageChunk)
        # Ensure hasattr checks work but values are falsy/None
        msg.tool_calls = []  # Empty list - falsy but exists
        msg.tool_call_chunks = []  # Empty list - falsy but exists
        msg.tool_call_id = "tool_id_123"  # This should be returned

        result = agent_manager._extract_tool_call_id_from_message(msg)
        assert result == "tool_id_123"

        # Test exception path (lines 511-513)
        # Create a message where accessing tool_calls raises an exception
        bad_msg = MagicMock()
        # Make the tool_calls attribute access raise an exception
        bad_msg.tool_calls = MagicMock()
        bad_msg.tool_calls.__getitem__ = MagicMock(side_effect=KeyError("No such key"))
        bad_msg.tool_calls.__bool__ = MagicMock(return_value=True)  # Make it truthy

        with patch("template_agent.src.core.manager.app_logger") as mock_logger:
            result = agent_manager._extract_tool_call_id_from_message(bad_msg)
            assert result is None
            # Verify debug was called with the error
            mock_logger.debug.assert_called_once()
            assert (
                "Could not extract tool call ID from message"
                in mock_logger.debug.call_args[0][0]
            )

    def test_update_tool_tracking_with_tool_message(self, agent_manager):
        """Test _update_tool_call_tracking with ToolMessage (lines 557-560)."""
        # Test tracking tool response ID (lines 557-560)
        # Create a message without tool_calls but with tool_call_id
        msg = MagicMock()
        # Make tool_calls falsy but existing
        msg.tool_calls = []  # Empty list - falsy
        msg.tool_call_id = "tool_resp_123"  # Has tool_call_id

        # Create metadata dict (required for messages mode)
        metadata = {}

        with patch("template_agent.src.core.manager.app_logger") as mock_logger:
            # Call with messages mode - expects (msg, metadata) tuple
            agent_manager._update_tool_call_tracking("messages", (msg, metadata))

            # Verify tool_call_id was tracked
            assert agent_manager._current_tool_call_id == "tool_resp_123"
            mock_logger.debug.assert_called_with(
                "Tracking tool response ID from message: tool_resp_123"
            )

    def test_update_tool_tracking_exception_handling(self, agent_manager):
        """Test _update_tool_call_tracking exception handling (lines 562-563)."""
        # Create an event that will cause an exception when iterating
        bad_event = MagicMock()
        # Make iteration fail
        bad_event.__iter__ = MagicMock(side_effect=Exception("Iteration error"))

        with patch("template_agent.src.core.manager.app_logger") as mock_logger:
            # Should not raise exception, just log it
            agent_manager._update_tool_call_tracking("messages", bad_event)

            # Verify error was logged
            mock_logger.debug.assert_called_with(
                "Error updating tool call tracking: Iteration error"
            )


class TestAgentManagerMissingCoverage:
    """Additional tests to achieve 100% coverage for AgentManager - testing specific missing lines."""

    @pytest.fixture
    def agent_manager(self):
        """Create an AgentManager instance."""
        return AgentManager(redhat_sso_token="test_token")

    @pytest.mark.asyncio
    async def test_stream_response_without_user_id_and_message(self, agent_manager):
        """Test stream_response when user_id and message are None (lines 80, 84)."""
        # Create request without user_id (message can't be None per schema)
        request = StreamRequest(
            message="",  # Empty string to test the else branch
            thread_id="thread_123",
            session_id="session_456",
            user_id=None,
            stream_tokens=True,
        )

        with patch(
            "template_agent.src.core.manager.get_template_agent"
        ) as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.astream = AsyncMock(return_value=AsyncMock())
            mock_agent.astream.return_value.__aiter__.return_value = []
            mock_get_agent.return_value.__aenter__.return_value = mock_agent

            # Patch _handle_input to return valid data
            with patch.object(agent_manager, "_handle_input") as mock_handle_input:
                mock_handle_input.return_value = (
                    {"input": {"messages": [{"content": "test", "type": "human"}]}},
                    "run_123",
                    "thread_123",
                )

                # Execute - should handle None user_id and message
                events = []
                async for event in agent_manager.stream_response(request):
                    events.append(event)

                # Verify get_template_agent was called with None for empty/missing values
                mock_get_agent.assert_called_once_with(
                    "test_token",
                    enable_checkpointing=True,
                    user_id=None,  # None because user_id was None
                    message=None,  # None because message was empty string
                )

    @pytest.mark.asyncio
    async def test_stream_response_non_tuple_event(self, agent_manager):
        """Test stream_response when event is not a tuple (line 109)."""
        request = StreamRequest(
            message="test",
            thread_id="thread_123",
            stream_tokens=True,
        )

        with patch(
            "template_agent.src.core.manager.get_template_agent"
        ) as mock_get_agent:
            mock_agent = AsyncMock()

            # Create a mock async iterator that yields non-tuple events
            async def mock_stream():
                yield "not_a_tuple"  # This should be skipped
                yield ("messages", [AIMessage(content="test response")])

            mock_agent.astream = MagicMock(return_value=mock_stream())
            mock_get_agent.return_value.__aenter__.return_value = mock_agent

            with patch.object(agent_manager, "_handle_input") as mock_handle_input:
                mock_handle_input.return_value = (
                    {"input": {"messages": [{"content": "test", "type": "human"}]}},
                    "run_123",
                    "thread_123",
                )

                # Execute
                events = []
                async for event in agent_manager.stream_response(request):
                    events.append(event)

                # Should have processed only the tuple event
                assert len(events) >= 1
