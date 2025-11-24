"""Tests for the prompt module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from template_agent.src.core.prompt import (
    get_current_date,
    get_system_prompt,
    get_user_preferences,
    get_contextual_memories,
)


class TestPrompt:
    """Test cases for prompt functions."""

    def test_get_current_date(self):
        """Test get_current_date returns formatted date string."""
        date_str = get_current_date()
        assert isinstance(date_str, str)
        # Should be in format "Month Day, Year" (e.g., "December 25, 2024")
        assert len(date_str.split()) == 3

    def test_get_system_prompt(self):
        """Test get_system_prompt returns non-empty string."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Template Agent" in prompt
        assert "Today's date is" in prompt

    @patch("template_agent.src.core.prompt.get_current_date")
    def test_get_system_prompt_includes_date(self, mock_get_date):
        """Test that get_system_prompt includes the current date."""
        mock_get_date.return_value = "December 25, 2024"
        prompt = get_system_prompt()
        assert "Today's date is December 25, 2024" in prompt

    @pytest.mark.asyncio
    async def test_get_user_preferences_empty(self):
        """Test get_user_preferences returns empty string when no preferences."""
        mock_store = AsyncMock()
        mock_store.asearch.return_value = []

        result = await get_user_preferences(mock_store, ("preferences", "user123"))

        assert result == ""
        mock_store.asearch.assert_called_once_with(("preferences", "user123"))

    @pytest.mark.asyncio
    async def test_get_user_preferences_with_data(self):
        """Test get_user_preferences returns formatted preferences."""
        mock_preference = MagicMock()
        mock_preference.key = "language"
        mock_preference.value = {"content": "Spanish"}

        mock_store = AsyncMock()
        mock_store.asearch.return_value = [mock_preference]

        result = await get_user_preferences(mock_store, ("preferences",))

        assert result == "\n\n**User Preferences:**\nlanguage: Spanish\n"
        mock_store.asearch.assert_called_once_with(("preferences",))

    @pytest.mark.asyncio
    async def test_get_user_preferences_multiple(self):
        """Test get_user_preferences with multiple preferences."""
        mock_pref1 = MagicMock()
        mock_pref1.key = "language"
        mock_pref1.value = {"content": "English"}

        mock_pref2 = MagicMock()
        mock_pref2.key = "theme"
        mock_pref2.value = {"content": "dark"}

        mock_store = AsyncMock()
        mock_store.asearch.return_value = [mock_pref1, mock_pref2]

        result = await get_user_preferences(mock_store, ("preferences", "user456"))

        assert "**User Preferences:**" in result
        assert "language: English" in result
        assert "theme: dark" in result

    @pytest.mark.asyncio
    async def test_get_contextual_memories_empty(self):
        """Test get_contextual_memories returns empty string when no memories."""
        mock_store = AsyncMock()
        mock_store.asearch.return_value = []

        result = await get_contextual_memories(
            mock_store, ("memory", "user123"), "test query"
        )

        assert result == ""
        mock_store.asearch.assert_called_once_with(
            ("memory", "user123"), query="test query"
        )

    @pytest.mark.asyncio
    async def test_get_contextual_memories_with_data(self):
        """Test get_contextual_memories returns formatted memories."""
        mock_memory = MagicMock()
        mock_memory.key = "favorite_ice_cream"
        mock_memory.value = {"content": "vanilla, mint"}

        mock_store = AsyncMock()
        mock_store.asearch.return_value = [mock_memory]

        result = await get_contextual_memories(
            mock_store, ("memory",), "ice cream preference"
        )

        assert (
            result
            == "\n\n**Contextual Memories:**\nfavorite_ice_cream: vanilla, mint\n"
        )
        mock_store.asearch.assert_called_once_with(
            ("memory",), query="ice cream preference"
        )

    @pytest.mark.asyncio
    async def test_get_contextual_memories_multiple(self):
        """Test get_contextual_memories with multiple memories."""
        mock_mem1 = MagicMock()
        mock_mem1.key = "favorite_color"
        mock_mem1.value = {"content": "blue"}

        mock_mem2 = MagicMock()
        mock_mem2.key = "favorite_food"
        mock_mem2.value = {"content": "pizza"}

        mock_store = AsyncMock()
        mock_store.asearch.return_value = [mock_mem1, mock_mem2]

        result = await get_contextual_memories(
            mock_store, ("memory", "user789"), "preferences"
        )

        assert "**Contextual Memories:**" in result
        assert "favorite_color: blue" in result
        assert "favorite_food: pizza" in result
