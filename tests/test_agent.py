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


class TestDatabaseInitialization:
    """Test cases for database initialization."""

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.AsyncPostgresSaver")
    @patch("template_agent.src.core.agent.logger")
    async def test_initialize_database_no_setup_method(
        self, mock_logger, mock_async_postgres_saver, mock_settings
    ):
        """Test database initialization when setup method doesn't exist (line 58)."""
        from template_agent.src.core.agent import initialize_database

        # Setup
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        # Mock the checkpoint without setup method
        mock_checkpoint = AsyncMock()
        delattr(mock_checkpoint, "setup")
        mock_async_postgres_saver.from_conn_string.return_value.__aenter__.return_value = mock_checkpoint

        # Execute
        await initialize_database()

        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "AsyncPostgresSaver does not have setup method - schema may need manual creation"
        )

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.logger")
    async def test_initialize_database_inmemory_mode(self, mock_logger, mock_settings):
        """Test database initialization skips when using in-memory storage (lines 45-46)."""
        from template_agent.src.core.agent import initialize_database

        # Setup
        mock_settings.USE_INMEMORY_SAVER = True

        # Execute
        await initialize_database()

        # Verify it logs and returns early
        mock_logger.info.assert_called_with(
            "Using in-memory storage - skipping database initialization"
        )

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.AsyncPostgresSaver")
    @patch("template_agent.src.core.agent.logger")
    async def test_initialize_database_with_setup_method(
        self, mock_logger, mock_async_postgres_saver, mock_settings
    ):
        """Test database initialization with setup method (lines 55-56)."""
        from template_agent.src.core.agent import initialize_database

        # Setup
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        # Mock the checkpoint with setup method
        mock_checkpoint = AsyncMock()
        mock_checkpoint.setup = AsyncMock()
        mock_async_postgres_saver.from_conn_string.return_value.__aenter__.return_value = mock_checkpoint

        # Execute
        await initialize_database()

        # Verify setup was called and success was logged
        mock_checkpoint.setup.assert_called_once()
        mock_logger.info.assert_any_call("Database schema initialized successfully")

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.AsyncPostgresSaver")
    @patch("template_agent.src.core.agent.logger")
    async def test_initialize_database_connection_error(
        self, mock_logger, mock_async_postgres_saver, mock_settings
    ):
        """Test database initialization error handling (lines 61-66)."""
        from template_agent.src.core.agent import initialize_database
        from template_agent.src.core.exceptions.exceptions import AppException

        # Setup
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        # Mock connection failure
        mock_async_postgres_saver.from_conn_string.side_effect = Exception(
            "Connection failed"
        )

        # Execute and verify exception
        with pytest.raises(AppException) as exc_info:
            await initialize_database()

        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert (
            "Failed to initialize database schema" in mock_logger.error.call_args[0][0]
        )

        # Verify exception message
        assert "Database initialization failed" in str(exc_info.value)


class TestMemoryToolsExceptionHandling:
    """Test cases for memory tools exception handling."""

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
    @patch("template_agent.src.core.agent.logger")
    async def test_memory_tools_creation_failure(
        self,
        mock_logger,
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
        """Test handling when memory tools creation fails (lines 126-130)."""
        # Setup settings
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True
        mock_settings.MCP_SERVER_NAME = "test_server"

        # Setup other mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Make memory tool creation fail
        mock_create_manage_memory_tool.side_effect = Exception(
            "Memory tool creation error"
        )

        # Execute
        async with get_template_agent(user_id="test_user"):
            pass

        # Verify error was logged and memory_tools is empty
        mock_logger.error.assert_any_call(
            "Error creating manage preference and contextual memory tools: Memory tool creation error"
        )
        # Agent should still be created, just without memory tools
        mock_create_react_agent.assert_called_once()
        call_kwargs = mock_create_react_agent.call_args.kwargs
        # tools should not include memory tools
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 0  # Only MCP tools (none in this case)


class TestMCPConnectionHandling:
    """Test cases for MCP connection handling."""

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.MultiServerMCPClient")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_global_checkpoint")
    @patch("template_agent.src.core.agent.get_global_memory_store")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    @patch("template_agent.src.core.agent.logger")
    async def test_mcp_ssl_verification_disabled(
        self,
        mock_logger,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_get_global_memory_store,
        mock_get_global_checkpoint,
        mock_create_react_agent,
        mock_mcp_client_class,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test MCP connection with SSL verification disabled (lines 157-158, 168)."""
        # Setup settings
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = False  # SSL verification disabled

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Mock MCP client - simulate successful connection
        mock_mcp_tools = [MagicMock(), MagicMock(), MagicMock()]
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_tools = AsyncMock(return_value=mock_mcp_tools)
        mock_mcp_client_class.return_value = mock_mcp_client

        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Execute
        async with get_template_agent(sso_token="test_token"):
            pass

        # Verify SSL warning was logged
        mock_logger.warning.assert_called_with(
            "SSL certificate verification disabled for MCP connection"
        )

        # Verify success log with tool count
        mock_logger.info.assert_any_call(
            f"Successfully connected to MCP server and loaded {len(mock_mcp_tools)} tools"
        )

        # Verify MCP client was configured with verify=False
        mock_mcp_client_class.assert_called_once()
        config = mock_mcp_client_class.call_args[0][0]["test_server"]
        assert config["verify"] is False

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("asyncio.wait_for")
    @patch("template_agent.src.core.agent.MultiServerMCPClient")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_global_checkpoint")
    @patch("template_agent.src.core.agent.get_global_memory_store")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    @patch("template_agent.src.core.agent.logger")
    async def test_mcp_connection_timeout_inmemory(
        self,
        mock_logger,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_get_global_memory_store,
        mock_get_global_checkpoint,
        mock_create_react_agent,
        mock_mcp_client_class,
        mock_wait_for,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test MCP connection timeout in local mode (lines 173-185)."""
        import asyncio as real_asyncio

        # Setup settings
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Mock the MCP client to return an async function that will be wrapped in wait_for
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_tools = AsyncMock()
        mock_mcp_client_class.return_value = mock_mcp_client

        # Mock asyncio.wait_for to raise TimeoutError
        mock_wait_for.side_effect = real_asyncio.TimeoutError()

        mock_checkpoint = MagicMock()
        mock_get_global_checkpoint.return_value = mock_checkpoint
        mock_store = AsyncMock()
        mock_get_global_memory_store.return_value = mock_store

        # Execute - should not raise in local mode
        async with get_template_agent():
            pass

        # Verify timeout error was logged
        expected_error_msg = (
            f"Timeout connecting to MCP server at {mock_settings.MCP_SERVER_URL} "
            f"after {mock_settings.MCP_CONNECTION_TIMEOUT}s. "
            f"Server may be down or unreachable."
        )
        mock_logger.error.assert_any_call(expected_error_msg)

        # Verify warning about local mode
        mock_logger.warning.assert_any_call(
            "Running in local development mode without MCP tools"
        )

        # Agent should still be created with empty tools
        mock_create_react_agent.assert_called_once()

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("asyncio.wait_for")
    @patch("template_agent.src.core.agent.MultiServerMCPClient")
    @patch("template_agent.src.core.agent.logger")
    async def test_mcp_connection_timeout_production(
        self,
        mock_logger,
        mock_mcp_client_class,
        mock_wait_for,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test MCP connection timeout in production mode raises error (lines 184-185)."""
        import asyncio as real_asyncio
        from template_agent.src.core.exceptions.exceptions import AppException

        # Setup settings for production
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.MCP_REQUIRED = True
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True
        mock_settings.database_uri = "postgresql://test"

        # Setup mocks
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Mock the MCP client
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_tools = AsyncMock()
        mock_mcp_client_class.return_value = mock_mcp_client

        # Mock asyncio.wait_for to raise TimeoutError
        mock_wait_for.side_effect = real_asyncio.TimeoutError()

        # Execute - should raise in production mode
        with pytest.raises(AppException) as exc_info:
            async with get_template_agent():
                pass

        # Verify the error message
        assert "Timeout connecting to MCP server" in str(exc_info.value)

        # Verify critical log
        expected_error_msg = (
            f"Timeout connecting to MCP server at {mock_settings.MCP_SERVER_URL} "
            f"after {mock_settings.MCP_CONNECTION_TIMEOUT}s. "
            f"Server may be down or unreachable."
        )
        mock_logger.critical.assert_called_with(expected_error_msg)

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("asyncio.wait_for")
    @patch("template_agent.src.core.agent.MultiServerMCPClient")
    @patch("template_agent.src.core.agent.logger")
    async def test_mcp_connection_error_production(
        self,
        mock_logger,
        mock_mcp_client_class,
        mock_wait_for,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test MCP connection error in production mode (lines 203-208)."""
        from template_agent.src.core.exceptions.exceptions import AppException

        # Setup settings for production
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.MCP_REQUIRED = True
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True
        mock_settings.database_uri = "postgresql://test"

        # Setup mocks
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Mock the MCP client
        mock_mcp_client = AsyncMock()
        mock_mcp_client.get_tools = AsyncMock()
        mock_mcp_client_class.return_value = mock_mcp_client

        # Mock MCP connection to fail with a different exception
        connection_error = ConnectionError("Connection refused")
        mock_wait_for.side_effect = connection_error

        # Execute - should raise AppException in production mode
        with pytest.raises(AppException) as exc_info:
            async with get_template_agent():
                pass

        # Verify error details were logged
        mock_logger.error.assert_any_call(
            f"Failed to connect to MCP server at {mock_settings.MCP_SERVER_URL}",
            exc_info=True,
        )
        mock_logger.error.assert_any_call(
            f"MCP connection error type: {type(connection_error).__name__}"
        )
        mock_logger.error.assert_any_call(
            f"MCP connection error details: {str(connection_error)}"
        )

        # Verify critical log
        expected_error_msg = (
            f"Failed to connect to required MCP server at {mock_settings.MCP_SERVER_URL}. "
            f"Error: {type(connection_error).__name__}: {str(connection_error)}"
        )
        mock_logger.critical.assert_called_with(expected_error_msg)

        # Verify exception message contains expected error
        assert expected_error_msg in str(exc_info.value)


class TestAgentWithoutCheckpointing:
    """Test cases for agent without checkpointing."""

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_system_prompt")
    @patch("template_agent.src.core.agent.logger")
    async def test_agent_without_checkpointing(
        self,
        mock_logger,
        mock_get_system_prompt,
        mock_create_react_agent,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test creating agent without checkpointing for streaming (lines 219-229)."""
        # Setup settings
        mock_settings.USE_INMEMORY_SAVER = True
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        mock_agent = MagicMock()
        mock_create_react_agent.return_value = mock_agent

        # Execute with checkpointing disabled
        async with get_template_agent(enable_checkpointing=False) as agent:
            assert agent == mock_agent

        # Verify logs
        mock_logger.info.assert_any_call(
            "Creating agent without checkpointing for streaming-only operations"
        )
        mock_logger.info.assert_any_call(
            "Template agent initialized successfully without checkpointing"
        )

        # Verify agent was created without checkpointer or store
        mock_create_react_agent.assert_called_once()
        call_kwargs = mock_create_react_agent.call_args.kwargs
        assert (
            "checkpointer" not in call_kwargs or call_kwargs.get("checkpointer") is None
        )
        assert "store" not in call_kwargs or call_kwargs.get("store") is None
        assert call_kwargs["prompt"] == "System prompt"


class TestPostgreSQLStorage:
    """Test cases for PostgreSQL storage."""

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("template_agent.src.core.agent.MultiServerMCPClient")
    @patch("asyncio.wait_for")
    @patch("template_agent.src.core.agent.AsyncPostgresSaver")
    @patch("template_agent.src.core.agent.AsyncPostgresStore")
    @patch("template_agent.src.core.agent.get_embedding_config")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    @patch("template_agent.src.core.agent.logger")
    async def test_postgresql_storage(
        self,
        mock_logger,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_create_react_agent,
        mock_get_embedding_config,
        mock_async_postgres_store_class,
        mock_async_postgres_saver_class,
        mock_wait_for,
        mock_mcp_client_class,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test agent with PostgreSQL storage (lines 252-281)."""
        # Setup settings for production with PostgreSQL
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True
        mock_settings.database_uri = "postgresql://user:pass@localhost/db"

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = (
            "\n\n**User Preferences:**\nlanguage: English\n"
        )
        mock_get_contextual_memories.return_value = (
            "\n\n**Contextual Memories:**\nname: John\n"
        )
        mock_get_embedding_config.return_value = {"embedding": "config"}

        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Mock MCP connection to return empty tools
        mock_mcp_tools = []
        mock_wait_for.return_value = mock_mcp_tools

        # Mock checkpoint and store
        mock_checkpoint = AsyncMock()
        mock_checkpoint.setup = AsyncMock()
        mock_async_postgres_saver_class.from_conn_string.return_value.__aenter__.return_value = mock_checkpoint

        mock_store = AsyncMock()
        mock_store.setup = AsyncMock()
        mock_async_postgres_store_class.from_conn_string.return_value.__aenter__.return_value = mock_store

        mock_agent = MagicMock()
        mock_create_react_agent.return_value = mock_agent

        # Execute
        async with get_template_agent(user_id="user123", message="Hello") as agent:
            assert agent == mock_agent

        # Verify PostgreSQL setup was called
        mock_async_postgres_saver_class.from_conn_string.assert_called_with(
            mock_settings.database_uri
        )
        mock_async_postgres_store_class.from_conn_string.assert_called_with(
            mock_settings.database_uri, index={"embedding": "config"}
        )

        # Verify setup methods were called
        mock_checkpoint.setup.assert_called_once()
        mock_store.setup.assert_called_once()

        # Verify preferences and memories were fetched
        mock_get_user_preferences.assert_called_with(
            mock_store, ("preferences", "user123")
        )
        mock_get_contextual_memories.assert_called_with(
            mock_store, ("memory", "user123"), "Hello"
        )

        # Verify agent was created with PostgreSQL storage
        mock_create_react_agent.assert_called_once()
        call_kwargs = mock_create_react_agent.call_args.kwargs
        assert call_kwargs["checkpointer"] == mock_checkpoint
        assert call_kwargs["store"] == mock_store
        expected_prompt = "System prompt\n\n**User Preferences:**\nlanguage: English\n\n\n**Contextual Memories:**\nname: John\n"
        assert call_kwargs["prompt"] == expected_prompt

        # Verify success log
        mock_logger.info.assert_any_call("Using PostgreSQL checkpoint for production")
        mock_logger.info.assert_any_call(
            "Template agent initialized successfully with PostgreSQL checkpoint"
        )

    @pytest.mark.asyncio
    @patch("template_agent.src.core.agent.ChatGoogleGenerativeAI")
    @patch("template_agent.src.core.agent.settings")
    @patch("template_agent.src.core.agent.create_manage_memory_tool")
    @patch("asyncio.wait_for")
    @patch("template_agent.src.core.agent.AsyncPostgresSaver")
    @patch("template_agent.src.core.agent.AsyncPostgresStore")
    @patch("template_agent.src.core.agent.get_embedding_config")
    @patch("template_agent.src.core.agent.create_react_agent")
    @patch("template_agent.src.core.agent.get_user_preferences")
    @patch("template_agent.src.core.agent.get_contextual_memories")
    @patch("template_agent.src.core.agent.get_system_prompt")
    async def test_postgresql_storage_no_setup_methods(
        self,
        mock_get_system_prompt,
        mock_get_contextual_memories,
        mock_get_user_preferences,
        mock_create_react_agent,
        mock_get_embedding_config,
        mock_async_postgres_store_class,
        mock_async_postgres_saver_class,
        mock_wait_for,
        mock_create_manage_memory_tool,
        mock_settings,
        mock_chat_model,
    ):
        """Test PostgreSQL storage when setup methods don't exist."""
        # Setup settings for production with PostgreSQL
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.MCP_REQUIRED = False
        mock_settings.MCP_CONNECTION_TIMEOUT = 10
        mock_settings.MCP_SERVER_URL = "http://localhost:5001/mcp/"
        mock_settings.MCP_SERVER_NAME = "test_server"
        mock_settings.MCP_TRANSPORT_PROTOCOL = "sse"
        mock_settings.MCP_SSL_VERIFY = True
        mock_settings.database_uri = "postgresql://user:pass@localhost/db"

        # Setup mocks
        mock_get_system_prompt.return_value = "System prompt"
        mock_get_user_preferences.return_value = ""
        mock_get_contextual_memories.return_value = ""
        mock_get_embedding_config.return_value = {"embedding": "config"}

        mock_pref_tool = MagicMock()
        mock_context_tool = MagicMock()
        mock_create_manage_memory_tool.side_effect = [mock_pref_tool, mock_context_tool]

        # Mock MCP connection to return empty tools
        mock_wait_for.return_value = []

        # Mock checkpoint and store WITHOUT setup methods
        mock_checkpoint = AsyncMock()
        delattr(mock_checkpoint, "setup")  # Remove setup method
        mock_async_postgres_saver_class.from_conn_string.return_value.__aenter__.return_value = mock_checkpoint

        mock_store = AsyncMock()
        delattr(mock_store, "setup")  # Remove setup method
        mock_async_postgres_store_class.from_conn_string.return_value.__aenter__.return_value = mock_store

        mock_agent = MagicMock()
        mock_create_react_agent.return_value = mock_agent

        # Execute - should not fail even without setup methods
        async with get_template_agent() as agent:
            assert agent == mock_agent

        # Verify agent was still created successfully
        mock_create_react_agent.assert_called_once()
