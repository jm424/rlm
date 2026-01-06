# RLM Utilities

from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.tool_prompts import (
    RLM_TOOL_SYSTEM_PROMPT,
    build_tool_rlm_messages,
    build_tool_system_prompt,
    build_tool_user_prompt,
)

__all__ = [
    # Standard RLM prompts
    "RLM_SYSTEM_PROMPT",
    "build_rlm_system_prompt",
    "build_user_prompt",
    # Tool-enabled RLM prompts
    "RLM_TOOL_SYSTEM_PROMPT",
    "build_tool_system_prompt",
    "build_tool_rlm_messages",
    "build_tool_user_prompt",
]

