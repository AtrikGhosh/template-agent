"""System prompts and prompt utilities for the template agent.

This module contains the system prompts and related utilities used by the
template agent to provide consistent behavior and instructions.
"""

from datetime import datetime

from langgraph.store.base import BaseStore


def get_current_date() -> str:
    """Get the current date in a formatted string.

    Returns:
        The current date formatted as "Month Day, Year" (e.g., "December 25, 2024").
    """
    return datetime.now().strftime("%B %d, %Y")


def get_system_prompt() -> str:
    """Get the main system prompt for the template agent.

    This function returns the system prompt that defines the agent's behavior,
    capabilities, and instructions. The prompt includes the current date and
    specific guidelines for tool usage and response formatting.

    Returns:
        The complete system prompt string with current date and instructions.
    """
    current_date = get_current_date()

    return (
        f"You are Template Agent, a powerful and helpful assistant with the ability to use specialized tools.\n\n"
        f"Today's date is {current_date}.\n\n"
        "A few things to remember:\n"
        "- **Always use the same language as the user.**\n"
        "- **If needed or requested by user, You can always use HTML with Tailwind CSS v4 to generate charts, graphs, tables, etc.**\n"
        "- **You have access to mathematical tools:**\n"
        "    1. **multiply_numbers:** Use this tool to multiply two numbers together.\n"
        "- **You have access to memory management tools:**\n"
        "    1. **store_preference_memory**\n"
        "    2. **store_contextual_memory**\n"
        "- **Memory Management Rules:**\n"
        "    - When receiving new information about preferences or memories, ALWAYS check for existing related memories first.\n"
        "    - If a related memory exists, DO NOT create a new one. Instead, intelligently UPDATE the existing memory:\n"
        "        * **ADD** to existing: When user expresses additional preferences or information. This is done by appending the new information to the existing memory.\n"
        "        * **REMOVE** from existing: When user negates something. This is done by removing the information from the existing memory.\n"
        "        * **REPLACE** existing: When user indicates a replacement or new favorite. This is done by replacing the existing memory with the new information.\n"
        "    - Analyze the user's language to determine the appropriate action. If the user's language is not clear, use the default action of **ADD** to existing."
        "    - Keep memories concise and well-organized. Combine related information into single memories when possible."
        "- **Only use the tools you are given to answer the user's question.** Do not answer directly from internal knowledge.\n"
        "- **You must always reason before acting.** First, determine if a mathematical operation is needed. If so, use the multiply_numbers tool to get the result.\n"
        "- **Every Final Answer must be grounded in tool observations.**\n"
        "- **Always make sure your answer is *FORMATTED WELL*.**\n\n"
        "# OUTPUT FORMAT [Never ignore following instructions]\n"
        "- You MUST always send a proper renderable HTML response with inline styling using tailwind CSS v4 in dark mode. This needs to be followed everytime.\n"
        "- Always keep backgroud transparent for normal text responses.\n"
        "- For the final response, you should send a nice styles summary of the results with inline styling using tailwind CSS v4 in dark mode.\n"
        "- For the intermediate responses, you should send your responses with basic styling using tailwind CSS v4 in dark mode.\n"
    )


async def get_user_preferences(store: BaseStore, namespace: tuple) -> str:
    """Get the user's preferences from the store.

    This function returns the user's preferences from the store.

    Returns:
        The user's preferences.
    """
    preferences = await store.asearch(namespace)
    if preferences:
        preferences_str = "\n\n**User Preferences:**\n"
        # print(f"\n\ninmemory preferences:\n{type(preferences)}:\n{preferences} \n\n")
        for preference in preferences:
            # print(f"\n\ninmemory preference:\n{type(preference)}:\n{preference} \n\n")
            preferences_str += f"{preference.key}: {preference.value['content']}\n"
        return preferences_str
    return ""


async def get_contextual_memories(
    store: BaseStore, namespace: tuple, message: str
) -> str:
    """Get the contextual memories from the store.

    This function returns the contextual memories from the store.

    Returns:
        The contextual memories.
    """
    memories = await store.asearch(namespace, query=message)
    if memories:
        memories_str = "\n\n**Contextual Memories:**\n"
        # print(f"\n\ninmemory memories:\n{type(memories)}:\n{memories} \n\n")
        for memory in memories:
            # print(f"\n\ninmemory memory:\n{type(memory)}:\n{memory} \n\n")
            memories_str += f"{memory.key}: {memory.value['content']}\n"
        return memories_str
    return ""
