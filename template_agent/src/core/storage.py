"""Global storage management for the template agent system.

This module provides a single global checkpoint instance that persists across
the entire application lifecycle when using in-memory storage mode.
"""

from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from template_agent.src.settings import settings
from template_agent.utils.pylogger import get_python_logger
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model from settings
embedding_model = HuggingFaceEmbeddings(
    model_name=settings.HF_EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': True}
)

logger = get_python_logger(settings.PYTHON_LOG_LEVEL)

# Global checkpoint instance - single instance for the entire application lifecycle
_global_checkpoint: Optional[InMemorySaver] = None

# Global memory store instance - single instance for the entire application lifecycle
# This is used to store user preferences and context
_global_memory_store: Optional[InMemoryStore] = None

# Global thread registry to track threads by user_id
_thread_registry: dict[str, set[str]] = {}

def get_embedding_config() -> dict:
    """Get the embedding configuration.

    Returns:
        The embedding configuration.
    """
    return {
        'dims': settings.HF_EMBEDDING_MODEL_DIMS,
        'embed': embedding_model,
        'fields': ['content'],
        'distance_type': 'cosine',
    }

def get_global_checkpoint() -> InMemorySaver:
    """Get the global in-memory checkpoint instance.

    This creates a single checkpoint instance that persists for the entire
    application lifecycle, ensuring all components use the same storage.
    The same instance serves as both checkpointer and store.

    Returns:
        The global InMemorySaver instance.
    """
    global _global_checkpoint
    if _global_checkpoint is None:
        _global_checkpoint = InMemorySaver()
        logger.info("Created global InMemorySaver checkpoint instance")
    return _global_checkpoint

def get_global_memory_store() -> InMemoryStore:
    """Get the global in-memory memory store instance.

    This creates a single memory store instance that persists for the entire
    application lifecycle, ensuring all components use the same storage.
    """
    global _global_memory_store
    if _global_memory_store is None:
        _global_memory_store = InMemoryStore(
            index=get_embedding_config()
        )
        logger.info("Created global InMemoryStore memory store instance")
    return _global_memory_store

def register_thread(user_id: str, thread_id: str) -> None:
    """Register a thread for a user.

    Args:
        user_id: The user ID
        thread_id: The thread ID to register
    """
    global _thread_registry
    if user_id not in _thread_registry:
        _thread_registry[user_id] = set()
    _thread_registry[user_id].add(thread_id)
    logger.info(f"Registered thread {thread_id} for user {user_id}")


def get_user_threads(user_id: str) -> list[str]:
    """Get all threads for a user.

    Args:
        user_id: The user ID

    Returns:
        List of thread IDs for the user
    """
    global _thread_registry
    threads = list(_thread_registry.get(user_id, set()))
    logger.info(f"Retrieved {len(threads)} threads for user {user_id}: {threads}")
    return threads


def reset_global_storage() -> None:
    """Reset the global checkpoint instance.

    This is useful for testing or when you want to clear all data.
    """
    global _global_checkpoint, _thread_registry
    _global_checkpoint = None
    _global_memory_store = None
    _thread_registry = {}
    logger.info("Reset global checkpoint instance and thread registry")


# Backward compatibility aliases
get_shared_checkpointer = get_global_checkpoint
get_shared_store = get_global_checkpoint
reset_shared_storage = reset_global_storage
