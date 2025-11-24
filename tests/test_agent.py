"""Tests for the agent module memory namespace functionality."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set environment variables to use in-memory saver before import
os.environ["USE_INMEMORY_SAVER"] = "true"
os.environ["MCP_REQUIRED"] = "false"

from template_agent.src.core.agent import get_template_agent


class TestAgentMemoryNamespace:
    """Test cases for agent memory namespace handling."""

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.get_global_checkpoint")
    @patch("template_agent.src.core.agent.get_global_memory_store")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    async def test_memory_namespace_with_user_id(
        self,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_create_react_agent,
        mock_get_global_memory_store,
        mock_get_global_checkpoint,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test that memory namespace includes user_id when provided."""
        # Setup settings mock
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Mock the memory tool creation
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Use the agent with user_id
        async with get_template_agent(user_id="test_user_123", message="test message"):
            pass

        # Verify memory tools were created with correct namespaces
        calls = mock_create_manage_memory_tool.call_args_list
        assert len(calls) == 2

        # Check preference memory tool namespace
        pref_call = calls[0]
        assert pref_call.kwargs["name"] == "store_preference_memory"
        assert pref_call.kwargs["namespace"] == ("preferences", "test_user_123")

        # Check contextual memory tool namespace
        context_call = calls[1]
        assert context_call.kwargs["name"] == "store_contextual_memory"
        assert context_call.kwargs["namespace"] == ("memory", "test_user_123")

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.get_global_checkpoint")
    @patch("template_agent.src.core.agent.get_global_memory_store")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    async def test_memory_namespace_without_user_id(
        self,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_create_react_agent,
        mock_get_global_memory_store,
        mock_get_global_checkpoint,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test that memory namespace uses default when user_id is not provided."""
        # Setup settings mock
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Mock the memory tool creation
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Use the agent without user_id
        async with get_template_agent(message="test query"):
            pass

        # Verify memory tools were created with correct namespaces
        calls = mock_create_manage_memory_tool.call_args_list
        assert len(calls) == 2

        # Check preference memory tool namespace (single element tuple)
        pref_call = calls[0]
        assert pref_call.kwargs["name"] == "store_preference_memory"
        assert pref_call.kwargs["namespace"] == ("preferences",)

        # Check contextual memory tool namespace (single element tuple)
        context_call = calls[1]
        assert context_call.kwargs["name"] == "store_contextual_memory"
        assert context_call.kwargs["namespace"] == ("memory",)

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.get_global_checkpoint")
    @patch("template_agent.src.core.agent.get_global_memory_store")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    async def test_prompt_includes_preferences_and_memories(
        self,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_create_react_agent,
        mock_get_global_memory_store,
        mock_get_global_checkpoint,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test that agent prompt includes user preferences and contextual memories."""
        # Setup settings mock
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = (
            "\n\n**User Preferences:**\nlanguage: Spanish\n"
        )
        mock_get_contextual_memories.return_value = (
            "\n\n**Contextual Memories:**\nfavorite_color: blue\n"
        )
        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Mock the memory tool creation
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Use the agent
        async with get_template_agent(
            user_id="user123", message="what's my favorite color"
        ):
            pass

        # Verify the prompt was constructed correctly
        mock_create_react_agent.assert_called_once()
        call_kwargs = mock_create_react_agent.call_args.kwargs
        expected_prompt = "System prompt\n\n**User Preferences:**\nlanguage: Spanish\n\n\n**Contextual Memories:**\nfavorite_color: blue\n"
        assert call_kwargs["prompt"] == expected_prompt

        # Verify get_contextual_memories was called with the message
        mock_get_contextual_memories.assert_called_with(
            mock_store, ("memory", "user123"), "what's my favorite color"
        )

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.get_global_checkpoint")
    @patch("template_agent.src.core.agent.get_global_memory_store")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    async def test_message_none_handled_gracefully(
        self,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_create_react_agent,
        mock_get_global_memory_store,
        mock_get_global_checkpoint,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test that None message is handled gracefully."""
        # Setup settings mock
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Mock the memory tool creation
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Use the agent with None message
        async with get_template_agent(user_id="user123", message=None):
            pass

        # Verify get_contextual_memories was called with empty string instead of None
        mock_get_contextual_memories.assert_called_with(
            mock_store, ("memory", "user123"), ""
        )
