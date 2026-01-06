"""
Parsing utilities for RLM trjaectories.
"""

import re

from rlm.core.types import REPLResult, RLMIteration


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def _extract_balanced_parens(text: str, start_pos: int) -> str | None:
    """Extract content between balanced parentheses starting at start_pos.
    
    Args:
        text: The full text
        start_pos: Position of the opening '('
        
    Returns:
        Content between the balanced parentheses, or None if unbalanced
    """
    if start_pos >= len(text) or text[start_pos] != '(':
        return None
    
    depth = 0
    start_content = start_pos + 1
    for i in range(start_pos, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[start_content:i]
    return None  # Unbalanced


def find_final_answer(text: str) -> tuple[str, str] | None:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return (type, content).
    Returns None if neither pattern is found.
    
    Uses balanced parenthesis matching to handle nested parens in the answer.
    """
    # Check for FINAL_VAR pattern first - must be at start of line
    final_var_match = re.search(r"^\s*FINAL_VAR\(", text, re.MULTILINE)
    if final_var_match:
        paren_pos = text.find('(', final_var_match.start())
        content = _extract_balanced_parens(text, paren_pos)
        if content is not None:
            return ("FINAL_VAR", content.strip())

    # Check for FINAL pattern - must be at start of line
    final_match = re.search(r"^\s*FINAL\(", text, re.MULTILINE)
    if final_match:
        paren_pos = text.find('(', final_match.start())
        content = _extract_balanced_parens(text, paren_pos)
        if content is not None:
            return ("FINAL", content.strip())

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, str | int | float | bool | list | dict | tuple):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger=None) -> str | None:
    """Check if response contains a final answer.
    
    Args:
        response: The LLM response text to check
        repl_env: The REPL environment with locals dict
        logger: Optional logger with log_tool_execution method
    
    Returns:
        The resolved final answer string, or None if no final answer found
    """
    result = find_final_answer(response)
    if result is None:
        return None

    answer_type, content = result

    if answer_type == "FINAL":
        return content
    elif answer_type == "FINAL_VAR":
        # Get the variable directly from the REPL environment
        try:
            # Strip spaces, quotes, and newlines from variable name
            variable_name = content.strip().strip('"').strip("'").strip("\n").strip("\r")

            # Check if variable exists in the REPL environment's locals
            if variable_name in repl_env.locals:
                variable_value = repl_env.locals[variable_name]
                return str(variable_value)
            else:
                error_msg = f"Variable '{variable_name}' not found in REPL environment"
                if logger and hasattr(logger, 'log_tool_execution'):
                    logger.log_tool_execution("FINAL_VAR", error_msg)
                return None
        except Exception as e:
            error_msg = f"Error retrieving variable '{content}': {str(e)}"
            if logger and hasattr(logger, 'log_tool_execution'):
                logger.log_tool_execution("FINAL_VAR", error_msg)
            return None

    return None


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
