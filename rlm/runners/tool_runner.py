#!/usr/bin/env python3
"""
RLM Tool Runner - Runs RLM with ToolREPL for letta-code integration.

This script is spawned by the TypeScript RLM provider and communicates
via JSON messages on stdin/stdout.

Usage:
    python -m rlm.runners.tool_runner

Protocol:
    Input (stdin, one JSON object):
    {
        "prompt": "...",
        "model": "claude-sonnet-4-20250514",
        "backend": "anthropic",
        "max_iterations": 30,
        "api_key": "..." (optional, uses env var if not provided)
    }

    Output (stdout, JSON lines):
    {"type": "progress", "iteration": N, "code": "...", "output": "..."}
    {"type": "tool_request", "id": "...", "name": "...", "args": {...}}
    {"type": "final", "answer": "...", "iterations": N, "success": true}
    {"type": "error", "message": "..."}
"""

import json
import os
import sys
import threading
from typing import Any

# .env loading is done after we receive the config with project_root
from dotenv import load_dotenv

from rlm.clients import get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.rlm import RLM
from rlm.core.types import RLMIteration
from rlm.environments.tool_repl import ToolREPL
from rlm.utils.tool_prompts import build_tool_system_prompt


def serialize_locals_preview(locals_dict: dict, max_len: int = 200) -> dict[str, str]:
    """Serialize locals dict with truncated previews for IPC."""
    result = {}
    for k, v in locals_dict.items():
        if k.startswith("_"):
            continue
        try:
            repr_str = repr(v)
            if len(repr_str) > max_len:
                result[k] = repr_str[:max_len] + "..."
            else:
                result[k] = repr_str
        except Exception:
            result[k] = f"<{type(v).__name__}>"
    return result


class IPCToolREPL(ToolREPL):
    """
    Extended ToolREPL that uses stdin/stdout for IPC with the TypeScript host.
    """

    def __init__(self, **kwargs):
        # Use real stdin/stdout for IPC
        super().__init__(
            ipc_input=sys.stdin,
            ipc_output=sys.stdout,
            **kwargs
        )
        # Keep a separate stderr for debug output
        self._debug_output = sys.stderr

    def debug(self, msg: str):
        """Write debug message to stderr."""
        self._debug_output.write(f"[DEBUG] {msg}\n")
        self._debug_output.flush()


def send_message(msg: dict[str, Any]):
    """Send a JSON message to stdout."""
    print(json.dumps(msg), flush=True)


def format_conversation_history(history: list[dict[str, str]]) -> str:
    """Format conversation history as a readable string for the prompt."""
    if not history:
        return ""
    
    parts = ["# Conversation History", 
             "The following is the history of your conversation with the user in this session:\n"]
    
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        role_label = "User" if role == "user" else "Assistant"
        parts.append(f"**{role_label}:** {content}\n")
    
    parts.append("---\n")
    return "\n".join(parts)


def run_rlm_with_tools(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run RLM with ToolREPL and return the result.

    The tool() calls in the REPL will be routed to the TypeScript host
    via stdout/stdin IPC.
    """
    # Load .env from the letta-code project root if provided
    project_root = config.get("project_root")
    if project_root:
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
    
    prompt = config.get("prompt", "")
    model = config.get("model", "claude-sonnet-4-20250514")
    backend = config.get("backend", "anthropic")
    max_iterations = config.get("max_iterations", 30)
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    tool_definitions = config.get("tool_definitions", [])
    conversation_history = config.get("conversation_history", [])

    if not prompt:
        return {"success": False, "error": "No prompt provided"}

    if not api_key:
        return {"success": False, "error": "No API key provided"}

    # Prepend conversation history to prompt for session context
    history_context = format_conversation_history(conversation_history)
    if history_context:
        prompt = f"{history_context}\n# Current Task\n{prompt}"

    # Build system prompt with tool definitions included
    system_prompt = build_tool_system_prompt(tool_definitions)

    # Track iteration count for final result
    iteration_count = 0

    def on_iteration(iteration_num: int, iteration: RLMIteration) -> None:
        """Emit progress event to TypeScript host."""
        nonlocal iteration_count
        iteration_count = iteration_num

        # Collect code block info with variables
        code_blocks_info = []
        all_locals: dict[str, Any] = {}
        for cb in iteration.code_blocks:
            code_blocks_info.append({
                "code_preview": cb.code[:300] if cb.code else "",
                "stdout": cb.result.stdout[:500] if cb.result.stdout else "",
                "stderr": cb.result.stderr[:200] if cb.result.stderr else "",
                "locals": serialize_locals_preview(cb.result.locals) if cb.result.locals else {},
            })
            if cb.result.locals:
                all_locals.update(cb.result.locals)

        send_message({
            "type": "progress",
            "iteration": iteration_num,
            "max_iterations": max_iterations,
            "response_preview": iteration.response[:500] if iteration.response else "",
            "code_blocks": code_blocks_info,
            "variables": serialize_locals_preview(all_locals),
            "has_final": iteration.final_answer is not None,
            "iteration_time": iteration.iteration_time,
            # Per-iteration token usage (context window size for this LLM call)
            "input_tokens": iteration.input_tokens,
            "output_tokens": iteration.output_tokens,
        })

    # Create RLM instance with ToolREPL environment
    rlm = RLM(
        backend=backend,
        backend_kwargs={
            "model_name": model,
            "api_key": api_key,
        },
        environment="tool",  # Use ToolREPL
        environment_kwargs={
            "tool_definitions": tool_definitions,
        },
        max_iterations=max_iterations,
        custom_system_prompt=system_prompt,
        verbose=False,
        on_iteration=on_iteration,
    )

    try:
        # Run the RLM completion
        result = rlm.completion(prompt)

        # Extract usage summary for propagation to letta-code
        usage_data = None
        if result.usage_summary:
            usage_data = result.usage_summary.to_dict()

        return {
            "success": True,
            "answer": result.response or "",
            "iterations": iteration_count,
            "usage": usage_data,
            "execution_time_ms": int(result.execution_time * 1000),
            "root_model": result.root_model,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }


def main():
    """Main entry point for the tool runner."""
    # Read config from stdin
    try:
        config_line = sys.stdin.readline()
        if not config_line:
            send_message({"type": "error", "message": "No config received"})
            sys.exit(1)

        config = json.loads(config_line.strip())
    except json.JSONDecodeError as e:
        send_message({"type": "error", "message": f"Invalid config JSON: {e}"})
        sys.exit(1)

    # Send ready message
    send_message({"type": "ready"})

    # Run RLM
    result = run_rlm_with_tools(config)

    # Send final result
    if result.get("success"):
        send_message({
            "type": "final",
            "answer": result.get("answer", ""),
            "iterations": result.get("iterations", 0),
            "success": True,
            "usage": result.get("usage"),
            "execution_time_ms": result.get("execution_time_ms"),
            "root_model": result.get("root_model"),
        })
    else:
        send_message({
            "type": "error",
            "message": result.get("error", "Unknown error"),
        })


if __name__ == "__main__":
    main()

