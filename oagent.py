#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ollama",
#     "typer",
#     "pathspec",
# ]
# ///

import json
import pathlib
import sys
import time
from typing import Annotated, Any

import ollama
import pydantic
import typer
import pathspec
import subprocess
import functools
import atexit

# DEFAULT_MODEL = "qwq"
DEFAULT_MODEL = "qwq-patched:latest"

EXTRA_SYSTEM_PROMPT = f"""
IMPORTANT:
- Shell calls are tool calls, so must be wrapped in XML+JSON.
- Tool calls are a complete XML tag around a JSON object.
- Write "RESULT: CONCLUSIVE" if you found the answer. Otherwise, write "RESULT: INCONCLUSIVE".
- CRUCIAL: ALWAYS conclude your answer with "RESULT: CONCLUSIVE" or "RESULT: INCONCLUSIVE" as your last message. If you don't do this, a child will die an unmerciful death.
- DO NOT rely on your memory; use the shell tool to search files, install packages, etc.
- Figure out what tools you need, emit the tool calls, and then exit as quick as possible.
- NEVER write markdown code blocks for shell; use tools instead.
- Current time: {time.strftime("%Y-%m-%d %H:%M:%S")}
""".strip()

class ToolCallResult(pydantic.BaseModel):
    tool_call: ollama.Message.ToolCall.Function
    result: Any

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def to_prompt(self) -> str:
        return f"<tool>\n<tool_call>{self.tool_call.model_dump_json()}</tool_call>\n<result>{json.dumps(self.result)}</result>\n</tool>"

_num_tool_calls = 0
_TOOLS = []
def tool(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _num_tool_calls
        _num_tool_calls += 1
        print(f"Tool call: {fn.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = fn(*args, **kwargs)
            print(f"Tool result: {fn.__name__} -> {result[:200]}{'...' if len(str(result)) > 200 else ''}")
            return result
        except Exception as e:
            error_msg = f"ERROR: Tool {fn.__name__} failed with exception: {str(e)}"
            print(f"Tool error: {error_msg}")
            return error_msg
    _TOOLS.append(wrapper)
    return wrapper


def expand_path(path: str) -> pathlib.Path:
    """Expand ~ and environment variables in path, then resolve to absolute path."""
    return pathlib.Path(path).expanduser().resolve()


class DockerContainer:
    def __init__(self):
        self.container_id = None
    
    def start(self):
        """Start a long-running Alpine container."""
        if self.container_id is None:
            result = subprocess.run(
                "docker run -d --rm --read-only -v {}:/workspace:ro --workdir /workspace alpine:latest tail -f /dev/null".format(pathlib.Path.cwd()),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stdout}")
            container_id = result.stdout.strip()
            # Verify container is running
            check = subprocess.run(
                f"docker ps -q --filter id={container_id}",
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            if check.returncode != 0 or not check.stdout.strip():
                raise RuntimeError(f"Container failed to start properly")
            self.container_id = container_id
    
    def stop(self):
        """Stop the container if it's running."""
        if self.container_id:
            subprocess.run(f"docker stop {self.container_id}", shell=True)
            self.container_id = None


# Global container instance
docker_container = DockerContainer()


@tool
def shell(command: str) -> str:
    """Execute a shell command and return its output. The command executes in
    a docker container with the current directory mounted as /workspace, as readonly.
    Other than that, you can run any command whatsoever, including installing
    packages, etc. It's alpine linux, so commands are in ash shell. The container
    stays running for the duration of the entire chat session.

    Current directory: /workspace
    
    Parameters:
        command: The shell command to execute
    
    Returns:
        str: Combined stdout and stderr output
    """
    try:
        # Ensure container is running
        docker_container.start()
        
        # Execute command in existing container
        restricted_cmd = f"docker exec {docker_container.container_id} /bin/sh -c '{command}'"
        result = subprocess.run(
            restricted_cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = result.stdout.strip()
        return f"{output}\n[Exit {result.returncode}]" if output else f"[Exit {result.returncode}]"
    except Exception as e:
        return f"Error executing command: {str(e)}"

app = typer.Typer()

def print_and_buffer(output_buffer: list[str], *lines: str) -> None:
    """Print lines immediately and add them to the buffer."""
    for line in lines:
        print(line)
        output_buffer.append(line)
    print()  # Extra newline for readability
    output_buffer.append("")


@app.command(name="")
def run(
    prompt: Annotated[str | None, typer.Argument(help="The prompt to send to the model. Defaults to stdin.")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="The primary model to use.")] = DEFAULT_MODEL,
    no_vim: Annotated[bool, typer.Option("--no-vim", help="Do not open the output in vim.")] = False,
) -> None:
    """Run the model with the given prompt."""
    if prompt is None and sys.stdin.isatty():
        typer.echo("No prompt provided and stdin is not a tty, exiting.")
        return
    
    messages = [{"role": "user", "content": f"{EXTRA_SYSTEM_PROMPT}\n\n{prompt or sys.stdin.read()}"}]
    start = time.time()
    output_buffer = []
    model_overrides = []
    
    while True:
        response = ollama.chat(
            model=(len(model_overrides) and model_overrides.pop() or model),
            messages=messages,
            stream=False,  # Not supported with tool use yet
            tools=_TOOLS,
        )
        
        print_and_buffer(output_buffer,
            "=== Response ===",
            response.message.content
        )
        
        if not response.message.tool_calls:
            break
            
        print_and_buffer(output_buffer, "=== Tool Calls ===")
        # Execute each tool call and append results
        for tool_call in response.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            
            print_and_buffer(output_buffer,
                f"Tool: {fn_name}",
                f"Args: {json.dumps(fn_args, indent=2)}"
            )
            
            try:
                # Find and call the tool
                tool_fn = next(t for t in _TOOLS if t.__name__ == fn_name)
                result = tool_fn(**fn_args)
            except Exception as e:
                result = f"ERROR: Tool {fn_name} failed with exception: {str(e)}"
                print_and_buffer(output_buffer, "Error:", result)
            else:
                print_and_buffer(output_buffer,
                    "Result:",
                    str(result)
                )
            
            # Format the result and add it to messages
            tool_result = ToolCallResult(tool_call=tool_call.function, result=result)
            messages.append({"role": "user", "content": tool_result.to_prompt()})
    
    end = time.time()
    print_and_buffer(output_buffer,
        "=== Summary ===",
        f"✅ {_num_tool_calls} tool calls, {end - start:.2f}s"
    )
    
    # Show complete output in vim
    if not no_vim:
        open_in_vim("\n".join(output_buffer))


def open_in_vim(text: str) -> None:
    """Open the given text in vim using a temporary file."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        subprocess.run(['vim', tmp_path])
    finally:
        os.unlink(tmp_path)  # Clean up temp file after vim closes


# Register cleanup handler
def cleanup():
    docker_container.stop()

atexit.register(cleanup)


if __name__ == "__main__":
    # Start container before running app
    docker_container.start()
    try:
        app()
    finally:
        docker_container.stop()