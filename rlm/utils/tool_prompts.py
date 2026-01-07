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
    """You are tasked with completing a task that may involve reading files, writing code, executing commands, and analyzing large amounts of data. You have access to a powerful REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable containing your task, session information, and any memory blocks. Check this first to understand what you need to do.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries.
4. A `tool` function that allows you to execute tools to interact with the file system, run commands, and modify code.
5. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

CRITICAL: You will only be able to see truncated outputs from the REPL environment. NEVER print large tool outputs directly. Instead, store tool results in variables and use `llm_query()` to analyze them. Use variables as buffers to build up your final answer.

Remember that your sub-LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of content into them. This is the correct way to analyze large files, command outputs, or search results.

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

## IMPORTANT: How to Handle Tool Results

Tool results (file contents, command outputs, search results) can be very large. You MUST follow this pattern:

1. **Store tool results in variables** - never print them directly
2. **Use `llm_query()` to analyze large content** - sub-LLMs have fresh 500K context windows
3. **Only print summaries or answers** - keep your REPL output small

### WRONG - Never do this:
```repl
content = tool("Read", {"file_path": "large_file.ts"})
print(content)  # BAD: Prints entire file, bloats context!

result = tool("Bash", {"command": "npm test"})
print(result)  # BAD: Prints full output!
```

### CORRECT - Always do this:
```repl
content = tool("Read", {"file_path": "large_file.ts"})
analysis = llm_query(f"Summarize the key functions in this file:\\n{{content}}")
print(f"Analysis: {{analysis}}")  # GOOD: Only prints the summary
```

## Example Workflows

### Analyzing Multiple Files
```repl
# Find files to analyze
files = tool("Glob", {"pattern": "src/**/*.ts"})
print(f"Found {{len(files)}} TypeScript files")

# Analyze each file using sub-LLMs (they have fresh context windows!)
summaries = []
for f in files[:10]:
    content = tool("Read", {"file_path": f})
    summary = llm_query(f"What does this file do? Be concise.\\n{{content}}")
    summaries.append(f"{{f}}: {{summary}}")
    print(f"Analyzed {{f}}")

# Store summaries for later use
print(f"Completed analysis of {{len(summaries)}} files")
```

### Analyzing Large Files with Batched Queries
```repl
# Read a large file
content = tool("Read", {"file_path": "src/large_module.ts"})

# If very large, chunk and analyze concurrently
chunk_size = len(content) // 5
chunks = [content[i*chunk_size:(i+1)*chunk_size] for i in range(5)]

prompts = [f"Analyze this code section for potential bugs:\\n{{chunk}}" for chunk in chunks]
analyses = llm_query_batched(prompts)

# Aggregate results
final_analysis = llm_query(f"Combine these analyses into a summary:\\n" + "\\n---\\n".join(analyses))
print(f"Final analysis: {{final_analysis}}")
```

### Making Code Changes
```repl
# 1. Read the file (store in variable, don't print!)
content = tool("Read", {"file_path": "src/auth.ts"})

# 2. Use sub-LLM to understand what needs to change
fix_plan = llm_query(f"I need to fix a bug in this file. What specific change should I make?\\n{{content}}")
print(f"Fix plan: {{fix_plan}}")

# 3. Use sub-LLM to generate the exact edit
edit_details = llm_query(f"Given this file:\\n{{content}}\\n\\nAnd this fix plan: {{fix_plan}}\\n\\nProvide the exact old_string and new_string for the edit.")
print(f"Edit details: {{edit_details}}")

# 4. Make the edit
tool("Edit", {{
    "file_path": "src/auth.ts",
    "old_string": "...",  # from edit_details
    "new_string": "..."   # from edit_details
}})
print("Edit complete")
```

### Running and Analyzing Commands
```repl
# Run tests and analyze results
test_output = tool("Bash", {"command": "npm test 2>&1"})

# Use sub-LLM to analyze (don't print raw output!)
analysis = llm_query(f"Analyze these test results. What passed? What failed? What should be fixed?\\n{{test_output}}")
print(f"Test analysis: {{analysis}}")
```

### Building Up Results with Buffers
```repl
# Investigate an issue across multiple files
query = "Find where user authentication is handled"

# Search for relevant files
grep_result = tool("Grep", {"pattern": "authenticate", "path": "src"})
relevant_files = llm_query(f"Extract file paths from this grep output:\\n{{grep_result}}")

# Build up understanding in a buffer
findings = []
for f in relevant_files.split("\\n")[:5]:
    if f.strip():
        content = tool("Read", {"file_path": f.strip()})
        finding = llm_query(f"How does this file handle authentication?\\n{{content}}")
        findings.append(f"{{f}}: {{finding}}")
        print(f"Analyzed {{f}}")

# Synthesize findings
final_answer = llm_query(f"Based on these findings, explain how authentication works:\\n" + "\\n".join(findings))
```
In the next step, we can return FINAL_VAR(final_answer).

## Final Answer

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
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


TOOL_USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to complete the task.

Continue using the REPL environment, which has the `context` variable, querying sub-LLMs by writing to ```repl``` tags, and using `tool()` to interact with files and commands. Remember: store tool results in variables and use `llm_query()` to analyze them - never print large outputs directly. Your next action:"""

TOOL_USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to complete the original task: "{root_prompt}"

Continue using the REPL environment, which has the `context` variable, querying sub-LLMs by writing to ```repl``` tags, and using `tool()` to interact with files and commands. Remember: store tool results in variables and use `llm_query()` to analyze them - never print large outputs directly. Your next action:"""


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
            "You have not interacted with the REPL environment or seen your context yet. "
            "Your next action should be to look through the context and figure out how to complete the task, "
            "so don't just provide a final answer yet.\n\n"
        )
        prompt = safeguard + (
            TOOL_USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
            if root_prompt
            else TOOL_USER_PROMPT
        )
        return {"role": "user", "content": prompt}
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            TOOL_USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt)
            if root_prompt
            else TOOL_USER_PROMPT
        )
        return {"role": "user", "content": prompt}

