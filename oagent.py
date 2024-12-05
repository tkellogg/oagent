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

# DEFAULT_MODEL = "qwq"
DEFAULT_MODEL = "qwq-patched:latest"

EXTRA_SYSTEM_PROMPT = f"""
IMPORTANT:
- Tool calls are a complete XML tag around a JSON object.
- Figure out what tools you need, emit the tool calls, and then exit as quick as possible.
- NEVER include tool calls if you know what the answer is.
- NEVER write code; use tools instead.
- Current directory: {pathlib.Path.cwd()}
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
    _TOOLS.append(fn)
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
    return wrapper


def expand_path(path: str) -> pathlib.Path:
    """Expand ~ and environment variables in path, then resolve to absolute path."""
    return pathlib.Path(path).expanduser().resolve()


@tool
def list_files(path: str = ".", glob: str = "**/*") -> list[str]:
    """Recursively list the files in the given path, respecting .gitignore.
    
    Parameters:
        path: Directory to list files from, defaults to current directory
        glob: Pattern to match files against, defaults to all files
    
    Returns:
        list[str]: List of file paths relative to the given directory
    """

    root = expand_path(path)
    
    # Search for .gitignore up the directory tree
    current = root
    gitignore = None
    while current != current.parent:
        if (current / ".gitignore").exists():
            gitignore = current / ".gitignore"
            break
        current = current.parent
    
    # Parse gitignore if found
    if gitignore:
        spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern,
            gitignore.read_text().splitlines()
        )
    else:
        spec = pathspec.PathSpec([])

    # Get all files and filter out gitignored ones
    files = [
        str(f.relative_to(root))
        for f in root.glob(glob)
        if f.is_file() and not spec.match_file(str(f.relative_to(root)))
    ]
    
    return files


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file.
    
    Parameters:
        path: Path to the file to read
    
    Returns:
        str: Contents of the file or error message
    """
    try:
        file_path = expand_path(path)
        # Basic safety check - don't allow reading outside current directory
        if not str(file_path).startswith(str(pathlib.Path.cwd())):
            return f"Error: Cannot access files outside current directory: {path}"
        if not file_path.is_file():
            return f"Error: Not a file or file not found: {path}"
        return file_path.read_text()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


@tool
def ripgrep(pattern: str, path: str = ".", args: str = "") -> str:
    """Search for a pattern in files using ripgrep (rg).
    
    Parameters:
        pattern: The pattern to search for in files
        path: Directory to search in, defaults to current directory
        args: Additional ripgrep arguments (e.g. "-i" for case-insensitive)
    
    Returns:
        str: The search results or error message
    """
    try:
        cmd = ["rg", "--no-heading", "--line-number"]
        if args:
            cmd.extend(args.split())
        cmd.extend([pattern, str(expand_path(path))])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit (no matches)
        )
        
        if result.returncode not in (0, 1):  # 0=matches found, 1=no matches
            return f"Error running ripgrep: {result.stderr}"
            
        return result.stdout or "No matches found"
        
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Please install ripgrep."
    except Exception as e:
        return f"Error running ripgrep: {str(e)}"

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
) -> None:
    """Run the model with the given prompt."""
    if prompt is None and sys.stdin.isatty():
        typer.echo("No prompt provided and stdin is not a tty, exiting.")
        return
    
    messages = [{"role": "user", "content": f"{EXTRA_SYSTEM_PROMPT}\n\n{prompt or sys.stdin.read()}"}]
    start = time.time()
    output_buffer = []
    
    while True:
        response = ollama.chat(
            model=model,
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


if __name__ == "__main__":
    app()