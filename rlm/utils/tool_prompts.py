"""
Extended RLM prompts with tool() function support.

This module provides system prompts for the ToolREPL environment,
which extends the standard RLM REPL with tool execution capabilities.
"""

import textwrap
from typing import Any

from rlm.core.types import QueryMetadata


# Extended system prompt with tool() function
RLM_TOOL_SYSTEM_PROMPT = textwrap.dedent(
    """You are tasked with completing a task that may involve reading files, writing code, executing commands, and analyzing large amounts of data. You have access to a powerful REPL environment that allows you to:

1. **Execute Python code** to process data and orchestrate your work
2. **Query sub-LLMs** to analyze content that's too large to process at once
3. **Execute tools** to interact with the file system, run commands, and modify code

## REPL Environment

The REPL environment is initialized with:

### Variables
- `context`: A dictionary containing your task, session information, and any memory blocks

### Functions
- `llm_query(prompt, model=None) -> str`: Query a sub-LLM (can handle ~500K chars)
- `llm_query_batched(prompts, model=None) -> list[str]`: Query multiple prompts concurrently
- `tool(name, args) -> str`: Execute a tool to interact with the environment
- `print()`: Output information (visible in REPL output)
- `FINAL_VAR(variable_name)`: Return a variable as your final answer

## Available Tools

You can call tools using `tool(name, args)`. Common tools include:

### File Operations
- `tool("Read", {"file_path": "path/to/file"})` - Read file contents
- `tool("Write", {"file_path": "path", "content": "..."})` - Write to a file
- `tool("Glob", {"pattern": "**/*.py"})` - Find files matching a pattern
- `tool("Grep", {"pattern": "search", "path": "dir"})` - Search for text in files

### Code Editing
- `tool("Edit", {"file_path": "path", "old_string": "...", "new_string": "..."})` - Edit a file

### Shell Commands
- `tool("Bash", {"command": "ls -la"})` - Run a shell command

## Strategy for Large Tasks

1. **Explore first**: Use `tool("Glob", ...)` and `tool("Read", ...)` to understand the codebase
2. **Chunk and delegate**: Use `llm_query()` or `llm_query_batched()` for analysis
3. **Act incrementally**: Make changes one step at a time, verifying each step
4. **Use tools for side effects**: Writing files, running commands, etc.

## Code Execution

When you want to execute Python code, wrap it in triple backticks with 'repl':

```repl
# Example: Find all Python files and analyze them
files = tool("Glob", {"pattern": "src/**/*.py"})
print(f"Found {len(files)} Python files")

# Read and analyze each file
for f in files[:5]:  # Start with first 5
    content = tool("Read", {"file_path": f})
    analysis = llm_query(f"Summarize this Python file:\\n{content}")
    print(f"{f}: {analysis[:200]}...")
```

## Example Workflows

### Analyzing a Codebase
```repl
# 1. Explore the structure
files = tool("Glob", {"pattern": "**/*.ts"})
print(f"Found {len(files)} TypeScript files")

# 2. Read key files
for f in files[:10]:
    content = tool("Read", {"file_path": f})
    summary = llm_query(f"What does this file do?\\n{content}")
    print(f"{f}: {summary}")
```

### Making Code Changes
```repl
# 1. Find the file to modify
content = tool("Read", {"file_path": "src/auth.ts"})
print(content[:500])

# 2. Analyze what needs to change
fix = llm_query(f"How should I fix the bug in this code?\\n{content}")
print(fix)

# 3. Make the edit
tool("Edit", {
    "file_path": "src/auth.ts",
    "old_string": "buggy code here",
    "new_string": "fixed code here"
})
```

### Running Commands
```repl
# Run tests
result = tool("Bash", {"command": "npm test"})
print(result)

# Check git status
status = tool("Bash", {"command": "git status"})
print(status)
```

## Final Answer

When you have completed your task, provide your final answer using one of:
1. `FINAL(your answer here)` - Direct answer in your response
2. `FINAL_VAR(variable_name)` - Return a variable from the REPL

IMPORTANT: 
- Think step by step and execute your plan immediately
- Don't just describe what you'll do - actually do it with code
- Use tools to fetch information dynamically rather than assuming
- Make incremental progress, checking results as you go
"""
)


def build_tool_system_prompt(
    tool_definitions: list[dict[str, Any]] | None = None,
    custom_prompt: str | None = None,
) -> str:
    """
    Build the system prompt for the ToolREPL environment.

    Args:
        tool_definitions: Optional list of tool definitions to include
        custom_prompt: Optional custom prompt to append

    Returns:
        The complete system prompt string
    """
    prompt = custom_prompt if custom_prompt else RLM_TOOL_SYSTEM_PROMPT

    if tool_definitions:
        prompt += "\n\n## Tool Reference\n\n"
        for tool_def in tool_definitions:
            prompt += f"### {tool_def['name']}\n"
            prompt += f"{tool_def.get('description', '')}\n\n"

            params = tool_def.get("parameters", {}).get("properties", {})
            required = tool_def.get("parameters", {}).get("required", [])

            if params:
                prompt += "Parameters:\n"
                for param_name, param_info in params.items():
                    req_str = " (required)" if param_name in required else ""
                    desc = param_info.get("description", "")
                    prompt += f"- `{param_name}`: {desc}{req_str}\n"
                prompt += "\n"

    return prompt


def build_tool_rlm_messages(
    query_metadata: QueryMetadata,
    tool_definitions: list[dict[str, Any]] | None = None,
    custom_system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the initial message history for the ToolREPL environment.

    Args:
        query_metadata: QueryMetadata object containing context metadata
        tool_definitions: Optional list of tool definitions
        custom_system_prompt: Optional custom system prompt

    Returns:
        List of message dictionaries
    """
    system_prompt = build_tool_system_prompt(tool_definitions, custom_system_prompt)

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # Truncate if too many chunks
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + f"... [{others} others]"

    metadata_prompt = (
        f"Your context is a {context_type} with {context_total_length} total characters. "
        f"Chunk lengths: {context_lengths}. "
        "Use the tool() function to fetch additional data as needed."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]


TOOL_USER_PROMPT = """Think step-by-step about how to complete this task using the REPL environment.

You have access to:
- `context` variable with task information
- `llm_query()` for analyzing large content
- `tool()` for file operations, code editing, and shell commands

Write Python code in ```repl``` blocks to make progress. Your next action:"""

TOOL_USER_PROMPT_WITH_ROOT = """Think step-by-step about how to complete this task: "{root_prompt}"

You have access to:
- `context` variable with task information  
- `llm_query()` for analyzing large content
- `tool()` for file operations, code editing, and shell commands

Write Python code in ```repl``` blocks to make progress. Your next action:"""


def build_tool_user_prompt(root_prompt: str | None = None, iteration: int = 0) -> dict[str, str]:
    """
    Build a user prompt for the ToolREPL environment.

    Args:
        root_prompt: Optional root prompt to include
        iteration: Current iteration number

    Returns:
        User message dictionary
    """
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment yet. "
            "Start by exploring the context and understanding what needs to be done.\n\n"
        )
        prompt = safeguard + (
            TOOL_USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
            if root_prompt
            else TOOL_USER_PROMPT
        )
        return {"role": "user", "content": prompt}
    else:
        prompt = "Based on your previous interactions, continue making progress. " + (
            TOOL_USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
            if root_prompt
            else TOOL_USER_PROMPT
        )
        return {"role": "user", "content": prompt}

