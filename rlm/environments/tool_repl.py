"""
ToolREPL - Extended REPL environment with tool() function for external tool execution.

This environment extends LocalREPL with the ability to call external tools via IPC.
It's designed to work with a TypeScript host process that executes tools and returns results.

IPC Protocol (JSON-line over stdin/stdout):
- Tool request:  {"type": "tool_request", "id": "...", "name": "...", "args": {...}}
- Tool result:   {"type": "tool_result", "id": "...", "result": "...", "status": "success"|"error"}
- Progress:      {"type": "progress", "iteration": N, "code": "...", "output": "..."}
- Final:         {"type": "final", "answer": "..."}
- Error:         {"type": "error", "message": "..."}
"""

import json
import sys
import uuid
from typing import Any

from rlm.environments.local_repl import LocalREPL, _SAFE_BUILTINS


class ToolREPL(LocalREPL):
    """
    Extended REPL environment with tool() function for calling external tools.

    The tool() function communicates with a host process via JSON messages on stdin/stdout.
    This allows the RLM to invoke tools like Read, Write, Bash, Edit, etc. that are
    executed by the host process (e.g., letta-code).
    """

    # Default threshold for auto-storing large tool outputs in variables
    DEFAULT_TOOL_OUTPUT_THRESHOLD = 5000

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        tool_output_threshold: int | None = None,  # Chars before auto-storing in variable
        ipc_input=None,  # For testing: override stdin
        ipc_output=None,  # For testing: override stdout
        **kwargs,
    ):
        # IPC channels - default to stdin/stdout for real use
        self._ipc_input = ipc_input or sys.stdin
        self._ipc_output = ipc_output or sys.stdout
        self._original_stdout = sys.stdout  # Keep reference for non-IPC output

        # Tool definitions (for documentation/validation)
        self.tool_definitions = tool_definitions or []

        # Tool output threshold: constructor arg > env var > default
        if tool_output_threshold is not None:
            self.tool_output_threshold = tool_output_threshold
        else:
            import os
            self.tool_output_threshold = int(
                os.environ.get("RLM_TOOL_OUTPUT_THRESHOLD", self.DEFAULT_TOOL_OUTPUT_THRESHOLD)
            )

        # Counter for auto-generated variable names
        self._tool_result_counter = 0

        # Track tool calls made during execution
        self._pending_tool_calls: list[dict[str, Any]] = []

        # Call parent constructor (which calls setup())
        super().__init__(
            lm_handler_address=lm_handler_address,
            context_payload=context_payload,
            setup_code=setup_code,
            **kwargs,
        )

    def setup(self):
        """Setup the environment with tool() function added to globals."""
        # Call parent setup first
        super().setup()

        # Add tool() function to globals
        self.globals["tool"] = self._tool

        # Add tool definitions to context if available
        if self.tool_definitions:
            self.globals["AVAILABLE_TOOLS"] = self.tool_definitions

    def _send_ipc_message(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the host process via IPC."""
        # Write to IPC output (not captured stdout)
        json_line = json.dumps(message)
        self._ipc_output.write(json_line + "\n")
        self._ipc_output.flush()

    def _receive_ipc_message(self) -> dict[str, Any]:
        """Receive a JSON message from the host process via IPC."""
        line = self._ipc_input.readline()
        if not line:
            raise RuntimeError("IPC channel closed unexpectedly")
        return json.loads(line.strip())

    def _tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Execute a tool via IPC with the host process.

        Args:
            name: The tool name (e.g., "Read", "Write", "Bash", "Edit", "Glob", "Grep")
            args: Dictionary of arguments to pass to the tool

        Returns:
            The tool's result as a string, or an error message if the tool failed.
            Large results (> tool_output_threshold) are automatically stored in a
            variable and a placeholder is returned to prevent context bloat.

        Example:
            content = tool("Read", {"file_path": "src/main.ts"})
            files = tool("Glob", {"pattern": "**/*.py"})
            tool("Write", {"file_path": "out.txt", "content": "Hello"})
        """
        if args is None:
            args = {}

        request_id = str(uuid.uuid4())

        # Track this tool call
        tool_call_record = {
            "id": request_id,
            "name": name,
            "args": args,
        }
        self._pending_tool_calls.append(tool_call_record)

        # Send tool request to host
        self._send_ipc_message({
            "type": "tool_request",
            "id": request_id,
            "name": name,
            "args": args,
        })

        # Wait for response from host
        while True:
            response = self._receive_ipc_message()

            # Check if this is our response
            if response.get("type") == "tool_result" and response.get("id") == request_id:
                status = response.get("status", "error")
                result = response.get("result", "")

                # Update tool call record with result
                tool_call_record["status"] = status
                tool_call_record["result"] = result

                if status == "error":
                    return f"Error: {result}"

                # Auto-store large outputs in variables to prevent context bloat
                if len(result) > self.tool_output_threshold:
                    return self._store_large_result(name, result)

                return result

            elif response.get("type") == "error":
                # Global error from host
                error_msg = response.get("message", "Unknown error from host")
                tool_call_record["status"] = "error"
                tool_call_record["result"] = error_msg
                return f"Error: {error_msg}"

            # Ignore other message types (might be progress updates, etc.)

    def _store_large_result(self, tool_name: str, result: str) -> str:
        """Store a large tool result in a variable and return a placeholder.

        This prevents large tool outputs from entering stdout/message history
        even if the model prints the result. The model should use llm_query()
        to analyze the stored variable.

        Args:
            tool_name: Name of the tool that produced the result
            result: The large result string to store

        Returns:
            A placeholder string indicating where the result is stored
        """
        self._tool_result_counter += 1
        var_name = f"_tool_result_{self._tool_result_counter}"

        # Store in locals so it's accessible in the REPL
        self.locals[var_name] = result

        # Return informative placeholder
        size_kb = len(result) / 1024
        return (
            f"[Large output ({size_kb:.1f}KB) stored in variable '{var_name}'. "
            f"Use llm_query() to analyze it, e.g.: "
            f"llm_query(f\"Analyze this {tool_name} output: {{{var_name}}}\")]"
        )

    def send_progress(self, iteration: int, code: str, output: str) -> None:
        """Send a progress update to the host process."""
        self._send_ipc_message({
            "type": "progress",
            "iteration": iteration,
            "code": code,
            "output": output,
        })

    def send_final(self, answer: str) -> None:
        """Send the final answer to the host process."""
        self._send_ipc_message({
            "type": "final",
            "answer": answer,
        })

    def send_error(self, message: str) -> None:
        """Send an error message to the host process."""
        self._send_ipc_message({
            "type": "error",
            "message": message,
        })

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Get all tool calls made during execution."""
        return self._pending_tool_calls.copy()

    def clear_tool_calls(self) -> None:
        """Clear the list of pending tool calls."""
        self._pending_tool_calls = []


# Standalone runner for use as subprocess
def run_tool_repl_server():
    """
    Run ToolREPL as a subprocess server.

    This function is called when the module is run directly. It:
    1. Reads initialization config from stdin
    2. Creates a ToolREPL instance
    3. Processes commands until exit

    Protocol:
    - Init:    {"type": "init", "context": {...}, "tool_definitions": [...]}
    - Execute: {"type": "execute", "code": "..."}
    - Exit:    {"type": "exit"}
    """
    import sys

    # Read init message
    init_line = sys.stdin.readline()
    if not init_line:
        sys.stderr.write("Error: No init message received\n")
        sys.exit(1)

    try:
        init_msg = json.loads(init_line.strip())
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error: Invalid init JSON: {e}\n")
        sys.exit(1)

    if init_msg.get("type") != "init":
        sys.stderr.write(f"Error: Expected init message, got {init_msg.get('type')}\n")
        sys.exit(1)

    # Create ToolREPL instance
    repl = ToolREPL(
        context_payload=init_msg.get("context"),
        tool_definitions=init_msg.get("tool_definitions", []),
        lm_handler_address=tuple(init_msg["lm_handler_address"])
        if init_msg.get("lm_handler_address")
        else None,
    )

    # Send ready message
    print(json.dumps({"type": "ready"}))
    sys.stdout.flush()

    # Process commands
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            msg = json.loads(line.strip())
            msg_type = msg.get("type")

            if msg_type == "execute":
                code = msg.get("code", "")
                result = repl.execute_code(code)

                # Send execution result
                print(json.dumps({
                    "type": "execution_result",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "locals": list(result.locals.keys()),  # Just send variable names
                    "execution_time": result.execution_time,
                    "tool_calls": repl.get_tool_calls(),
                }))
                sys.stdout.flush()

                # Clear tool calls for next execution
                repl.clear_tool_calls()

            elif msg_type == "exit":
                repl.cleanup()
                break

            else:
                print(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))
                sys.stdout.flush()

        except json.JSONDecodeError as e:
            print(json.dumps({
                "type": "error",
                "message": f"Invalid JSON: {e}",
            }))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({
                "type": "error",
                "message": f"Error: {e}",
            }))
            sys.stdout.flush()


if __name__ == "__main__":
    run_tool_repl_server()

