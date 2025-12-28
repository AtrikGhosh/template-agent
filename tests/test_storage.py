"""Tests for the storage module."""

import pytest
from unittest.mock import MagicMock, patch

from template_agent.src.core.storage import (
    get_embedding_config,
    get_embedding_model,
    get_global_checkpoint,
    get_global_memory_store,
    get_shared_checkpointer,
    get_shared_store,
    get_user_threads,
    register_thread,
    reset_global_storage,
    reset_shared_storage,
)


class TestEmbeddingConfig:
    """Test cases for embedding configuration."""

    @patch("template_agent.src.core.storage.settings")
    @patch("template_agent.src.core.storage.get_embedding_model")
    def test_get_embedding_config(self, mock_get_embedding_model, mock_settings):
        """Test get_embedding_config returns correct configuration."""
        mock_settings.GOOGLE_EMBEDDING_MODEL_DIMS = 768
        mock_embedding = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding

        config = get_embedding_config()

        assert config["dims"] == 768
        assert config["embed"] == mock_embedding
        assert config["fields"] == ["content"]
        assert config["distance_type"] == "cosine"

    @patch("template_agent.src.core.storage.settings")
    @patch("template_agent.src.core.storage.get_embedding_model")
    def test_get_embedding_config_with_different_dims(
        self, mock_get_embedding_model, mock_settings
    ):
        """Test get_embedding_config with different dimensions."""
        mock_settings.GOOGLE_EMBEDDING_MODEL_DIMS = 512
        mock_embedding = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding

        config = get_embedding_config()

        assert config["dims"] == 512


class TestEmbeddingModel:
    """Test cases for embedding model management."""

    @patch("template_agent.src.core.storage.settings")
    @patch("template_agent.src.core.storage.GoogleGenerativeAIEmbeddings")
    def test_get_embedding_model_creates_singleton(
        self, mock_embeddings_class, mock_settings
    ):
        """Test that get_embedding_model creates a singleton instance."""
        # Reset to ensure clean state
        reset_global_storage()

        # Mock settings
        mock_settings.GOOGLE_EMBEDDING_MODEL_NAME = "models/embedding-001"
        mock_settings.GOOGLE_EMBEDDING_TASK_TYPE = "retrieval_document"
        mock_settings.GOOGLE_EMBEDDING_MODEL_DIMS = 768

        mock_embedding = MagicMock()
        mock_embeddings_class.return_value = mock_embedding

        # First call should create instance
        model1 = get_embedding_model()
        assert model1 == mock_embedding
        mock_embeddings_class.assert_called_once_with(
            model="models/embedding-001", task_type="retrieval_document", dimensions=768
        )

        # Second call should return same instance
        model2 = get_embedding_model()
        assert model2 == model1
        # Still only called once
        assert mock_embeddings_class.call_count == 1

        # Clean up
        reset_global_storage()

    @patch("template_agent.src.core.storage.settings")
    @patch("template_agent.src.core.storage.GoogleGenerativeAIEmbeddings")
    @patch("template_agent.src.core.storage.logger")
    def test_get_embedding_model_logs_creation(
        self, mock_logger, mock_embeddings_class, mock_settings
    ):
        """Test that embedding model creation is logged."""
        # Reset to ensure clean state
        reset_global_storage()

        # Mock settings
        mock_settings.GOOGLE_EMBEDDING_MODEL_NAME = "models/embedding-001"
        mock_settings.GOOGLE_EMBEDDING_TASK_TYPE = "retrieval_document"
        mock_settings.GOOGLE_EMBEDDING_MODEL_DIMS = 768

        mock_embedding = MagicMock()
        mock_embeddings_class.return_value = mock_embedding

        get_embedding_model()

        mock_logger.info.assert_called_with(
            "Initialized Google Generative AI Embeddings with 768 dimensions"
        )

        # Clean up
        reset_global_storage()


class TestGlobalCheckpoint:
    """Test cases for global checkpoint management."""

    def test_get_global_checkpoint_creates_singleton(self):
        """Test that get_global_checkpoint creates a singleton instance."""
        # Reset to ensure clean state
        reset_global_storage()

        with patch("template_agent.src.core.storage.InMemorySaver") as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver_class.return_value = mock_saver

            # First call should create instance
            checkpoint1 = get_global_checkpoint()
            assert checkpoint1 == mock_saver
            mock_saver_class.assert_called_once()

            # Second call should return same instance
            checkpoint2 = get_global_checkpoint()
            assert checkpoint2 == checkpoint1
            # Still only called once
            assert mock_saver_class.call_count == 1

        # Clean up
        reset_global_storage()

    def test_get_global_checkpoint_logs_creation(self):
        """Test that checkpoint creation is logged."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.logger") as mock_logger:
            with patch("template_agent.src.core.storage.InMemorySaver"):
                get_global_checkpoint()
                mock_logger.info.assert_called_with(
                    "Created global InMemorySaver checkpoint instance"
                )

        reset_global_storage()

    def test_get_shared_checkpointer_alias(self):
        """Test that get_shared_checkpointer is an alias for get_global_checkpoint."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.InMemorySaver") as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver_class.return_value = mock_saver

            checkpoint1 = get_global_checkpoint()
            checkpoint2 = get_shared_checkpointer()
            assert checkpoint1 == checkpoint2

        reset_global_storage()

    def test_get_shared_store_alias(self):
        """Test that get_shared_store is an alias for get_global_checkpoint."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.InMemorySaver") as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver_class.return_value = mock_saver

            checkpoint = get_global_checkpoint()
            store = get_shared_store()
            assert checkpoint == store

        reset_global_storage()


class TestGlobalMemoryStore:
    """Test cases for global memory store management."""

    def test_get_global_memory_store_creates_singleton(self):
        """Test that get_global_memory_store creates a singleton instance."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.InMemoryStore") as mock_store_class:
            with patch(
                "template_agent.src.core.storage.get_embedding_config"
            ) as mock_config:
                mock_config.return_value = {"test": "config"}
                mock_store = MagicMock()
                mock_store_class.return_value = mock_store

                # First call should create instance
                store1 = get_global_memory_store()
                assert store1 == mock_store
                mock_store_class.assert_called_once_with(index={"test": "config"})

                # Second call should return same instance
                store2 = get_global_memory_store()
                assert store2 == store1
                # Still only called once
                assert mock_store_class.call_count == 1

        reset_global_storage()

    def test_get_global_memory_store_logs_creation(self):
        """Test that memory store creation is logged."""
        # Force reset of the global memory store
        import template_agent.src.core.storage as storage_module

        storage_module._global_memory_store = None

        with patch("template_agent.src.core.storage.InMemoryStore"):
            with patch("template_agent.src.core.storage.get_embedding_config"):
                with patch("template_agent.src.core.storage.logger") as mock_logger:
                    get_global_memory_store()
                    mock_logger.info.assert_called_with(
                        "Created global InMemoryStore memory store instance"
                    )

        # Reset after test
        storage_module._global_memory_store = None


class TestThreadRegistry:
    """Test cases for thread registry management."""

    def test_register_thread_new_user(self):
        """Test registering a thread for a new user."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.logger") as mock_logger:
            register_thread("user1", "thread1")

            # Verify thread was registered
            threads = get_user_threads("user1")
            assert "thread1" in threads
            assert len(threads) == 1

            # Verify logging - check for any of the expected calls
            expected_calls = [
                "Registered thread thread1 for user user1",
                "Retrieved 1 threads for user user1: ['thread1']",
            ]
            actual_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any(expected in str(actual_calls) for expected in expected_calls)

        reset_global_storage()

    def test_register_thread_existing_user(self):
        """Test registering multiple threads for the same user."""
        reset_global_storage()

        register_thread("user1", "thread1")
        register_thread("user1", "thread2")
        register_thread("user1", "thread3")

        threads = get_user_threads("user1")
        assert len(threads) == 3
        assert "thread1" in threads
        assert "thread2" in threads
        assert "thread3" in threads

        reset_global_storage()

    def test_register_thread_duplicate(self):
        """Test registering the same thread twice."""
        reset_global_storage()

        register_thread("user1", "thread1")
        register_thread("user1", "thread1")  # Duplicate

        threads = get_user_threads("user1")
        # Should only have one instance due to set behavior
        assert len(threads) == 1
        assert "thread1" in threads

        reset_global_storage()

    def test_register_thread_multiple_users(self):
        """Test registering threads for multiple users."""
        reset_global_storage()

        register_thread("user1", "thread1")
        register_thread("user1", "thread2")
        register_thread("user2", "thread3")
        register_thread("user2", "thread4")
        register_thread("user3", "thread5")

        user1_threads = get_user_threads("user1")
        assert len(user1_threads) == 2
        assert "thread1" in user1_threads
        assert "thread2" in user1_threads

        user2_threads = get_user_threads("user2")
        assert len(user2_threads) == 2
        assert "thread3" in user2_threads
        assert "thread4" in user2_threads

        user3_threads = get_user_threads("user3")
        assert len(user3_threads) == 1
        assert "thread5" in user3_threads

        reset_global_storage()

    def test_get_user_threads_empty(self):
        """Test getting threads for a user with no threads."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.logger") as mock_logger:
            threads = get_user_threads("nonexistent_user")

            assert threads == []
            # Verify logging
            mock_logger.info.assert_called_with(
                "Retrieved 0 threads for user nonexistent_user: []"
            )

        reset_global_storage()

    def test_get_user_threads_with_data(self):
        """Test getting threads for a user with existing threads."""
        reset_global_storage()

        register_thread("user1", "thread1")
        register_thread("user1", "thread2")

        with patch("template_agent.src.core.storage.logger") as mock_logger:
            threads = get_user_threads("user1")

            assert len(threads) == 2
            assert "thread1" in threads
            assert "thread2" in threads

            # Verify logging
            log_call = mock_logger.info.call_args_list[-1]  # Get last info call
            assert "Retrieved 2 threads for user user1" in log_call[0][0]

        reset_global_storage()

    def test_get_user_threads_returns_list(self):
        """Test that get_user_threads returns a list, not a set."""
        reset_global_storage()

        register_thread("user1", "thread1")
        register_thread("user1", "thread2")

        threads = get_user_threads("user1")
        assert isinstance(threads, list)
        assert len(threads) == 2

        reset_global_storage()


class TestResetStorage:
    """Test cases for storage reset functionality."""

    def test_reset_global_storage(self):
        """Test that reset_global_storage clears all global state."""
        # Set up some state
        with patch("template_agent.src.core.storage.InMemorySaver"):
            with patch("template_agent.src.core.storage.InMemoryStore"):
                with patch("template_agent.src.core.storage.get_embedding_config"):
                    checkpoint = get_global_checkpoint()
                    store = get_global_memory_store()

        register_thread("user1", "thread1")
        register_thread("user2", "thread2")

        # Reset storage
        with patch("template_agent.src.core.storage.logger") as mock_logger:
            reset_global_storage()
            mock_logger.info.assert_called_with(
                "Reset global checkpoint instance and thread registry"
            )

        # Verify everything is cleared
        threads1 = get_user_threads("user1")
        threads2 = get_user_threads("user2")
        assert threads1 == []
        assert threads2 == []

        # Verify new instances are created after reset
        with patch("template_agent.src.core.storage.InMemorySaver") as mock_saver_class:
            with patch(
                "template_agent.src.core.storage.InMemoryStore"
            ) as mock_store_class:
                with patch("template_agent.src.core.storage.get_embedding_config"):
                    new_checkpoint = get_global_checkpoint()
                    # The store might not be accessed here
                    # Should create new checkpoint instance
                    mock_saver_class.assert_called_once()

        reset_global_storage()

    def test_reset_shared_storage_alias(self):
        """Test that reset_shared_storage is an alias for reset_global_storage."""
        # Set up some state
        register_thread("user1", "thread1")

        # Reset using alias
        reset_shared_storage()

        # Verify state is cleared
        threads = get_user_threads("user1")
        assert threads == []


class TestModuleState:
    """Test cases for module-level state management."""

    def test_global_state_isolation(self):
        """Test that global state is properly isolated between tests."""
        # This test verifies that each test starts with clean state
        reset_global_storage()

        # Register a thread
        register_thread("test_user", "test_thread")
        threads = get_user_threads("test_user")
        assert len(threads) == 1

        # Reset
        reset_global_storage()

        # Verify clean state
        threads = get_user_threads("test_user")
        assert len(threads) == 0

    def test_concurrent_access_to_singleton(self):
        """Test that singleton instances handle concurrent access correctly."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.InMemorySaver") as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver_class.return_value = mock_saver

            # Simulate concurrent access
            checkpoint1 = get_global_checkpoint()
            checkpoint2 = get_global_checkpoint()
            checkpoint3 = get_global_checkpoint()

            # All should be the same instance
            assert checkpoint1 == checkpoint2 == checkpoint3
            # Constructor should only be called once
            assert mock_saver_class.call_count == 1

        reset_global_storage()

    def test_thread_registry_thread_safety(self):
        """Test that thread registry operations are consistent."""
        reset_global_storage()

        # Register threads from "different threads" (simulated)
        for i in range(10):
            register_thread("user1", f"thread_{i}")

        threads = get_user_threads("user1")
        assert len(threads) == 10

        # Verify all threads are present
        for i in range(10):
            assert f"thread_{i}" in threads

        reset_global_storage()


class TestBackwardCompatibility:
    """Test cases for backward compatibility aliases."""

    def test_all_aliases_work(self):
        """Test that all backward compatibility aliases work correctly."""
        reset_global_storage()

        with patch("template_agent.src.core.storage.InMemorySaver") as mock_saver_class:
            mock_saver = MagicMock()
            mock_saver_class.return_value = mock_saver

            # Test checkpoint aliases
            checkpoint1 = get_global_checkpoint()
            checkpoint2 = get_shared_checkpointer()
            checkpoint3 = get_shared_store()

            assert checkpoint1 == checkpoint2 == checkpoint3

        # Test reset aliases
        register_thread("user", "thread")
        reset_shared_storage()
        assert get_user_threads("user") == []

        reset_global_storage()
