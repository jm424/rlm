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
from rlm.environments.tool_repl import ToolREPL
from rlm.utils.tool_prompts import RLM_TOOL_SYSTEM_PROMPT


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

    if not prompt:
        return {"success": False, "error": "No prompt provided"}

    if not api_key:
        return {"success": False, "error": "No API key provided"}

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
        custom_system_prompt=RLM_TOOL_SYSTEM_PROMPT,
        verbose=False,
    )

    try:
        # Run the RLM completion
        result = rlm.completion(prompt)

        return {
            "success": True,
            "answer": result.response or "",
            "iterations": 0,  # RLMChatCompletion doesn't track iterations directly
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
        })
    else:
        send_message({
            "type": "error",
            "message": result.get("error", "Unknown error"),
        })


if __name__ == "__main__":
    main()

