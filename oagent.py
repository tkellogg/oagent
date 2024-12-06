#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ollama",
#     "tiktoken",
#     "typer",
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
import subprocess
import functools
import atexit
import tiktoken

# DEFAULT_MODEL = "qwq"
# DEFAULT_MODEL = "qwq-patched:latest"
DEFAULT_MODEL = "qwen2.5:14b"
WISE_MODEL = "qwq-patched:latest"

EXTRA_SYSTEM_PROMPT = f"""
IMPORTANT:
- Shell calls are tool calls, so must be wrapped in XML+JSON.
- Tool calls are a complete XML tag around a JSON object.
- Write "RESULT: CONCLUSIVE" if you found the answer. Otherwise, write "RESULT: INCONCLUSIVE".
- CRUCIAL: ALWAYS conclude your answer with "RESULT: CONCLUSIVE" or "RESULT: INCONCLUSIVE" as your last message. If you don't do this, a child will die an unmerciful death.
- DO NOT rely on your memory; use the shell tool to search files, install packages, etc.
- Figure out what tools you need, emit the tool calls, and then exit as quick as possible.
- NEVER write markdown code blocks for shell; use tools instead.
- CONSIDER running `find .` or the project_planner tool before writing code.
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
def tool(fn_or_class):
    """Decorator that registers a function or class as a tool.
    For classes, it wraps the __call__ method and uses the class.name as the function name.
    """
    def make_wrapper(func, name):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _num_tool_calls
            _num_tool_calls += 1
            print(f"Tool call: {name}(args={args}, kwargs={kwargs})")
            try:
                result = func(*args, **kwargs)
                print(f"Tool result: {name} -> {result[:200]}{'...' if len(str(result)) > 200 else ''}")
                return result
            except Exception as e:
                error_msg = f"ERROR: Tool {name} failed with exception: {str(e)}"
                print(f"Tool error: {error_msg}")
                return error_msg
        return wrapper

    if isinstance(fn_or_class, type):  # If decorating a class
        cls = fn_or_class
        call_method = cls.__call__
        # Get signature without 'self'
        import inspect
        sig = inspect.signature(call_method)
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        
        def class_caller(*args, **kwargs):
            instance = cls(*args[:1])  # Create instance with first arg (context)
            return instance(**kwargs)  # Call with remaining kwargs
        
        wrapper = make_wrapper(class_caller, cls.name)
        
        # Set the name and signature for ollama
        wrapper.__name__ = cls.name
        wrapper.__annotations__ = {
            p.name: p.annotation
            for p in params
        }
        if sig.return_annotation is not inspect.Signature.empty:
            wrapper.__annotations__['return'] = sig.return_annotation
        
        # Mark as class-based tool
        wrapper._is_class_tool = True
        wrapper._tool_class = cls
        
        _TOOLS.append(wrapper)
        return cls
    
    else:  # If decorating a function
        wrapper = make_wrapper(fn_or_class, fn_or_class.__name__)
        wrapper._is_class_tool = False
        _TOOLS.append(wrapper)
        return wrapper


class ToolContext:
    """Context object passed to tools, containing conversation history and other metadata."""
    def __init__(self, messages: list[dict], tool_calls: list) -> None:
        self.messages = messages
        self.tool_calls = tool_calls  # Reference to shared tool calls list
        self.force_stop = False  # Whether to force stop the conversation


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


@tool
class ProjectPlanner:
    name = "project_planner"
    def __init__(self, context: ToolContext) -> None:
        self.context = context  # Keep reference to modify tool calls

    def __call__(self, question: str) -> str:
        """Ask for guidance on how to solve a problem. This tool is crucial for
        planning your next steps.
        IMPORTANT: Always use this tool before making other tool calls.
        
        Parameters:
            question: The question for the project planner to solve
        
        Returns:
            str: The project planner's analysis and suggestions
        """
        try:
            response = ollama.chat(
                model=WISE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly technical project planner. Analyze the conversation, "
                                "identify core problems, and suggest specific solutions "
                                "or next steps. Focus on what tools or approaches "
                                "could help solve the problem."
                    },
                    *self.context.messages,
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                stream=False,
                tools=_TOOLS,
            )
            # Add planner's response to messages
            self.context.messages.append({"role": "assistant", "content": response.message.content})
            # Add any tool calls to the context
            self.context.tool_calls.extend(response.message.tool_calls or [])
            return response.message.content
        except Exception as e:
            return f"Error consulting wise model: {str(e)}"


@tool
class Judge:
    name = "judge_completion"
    def __init__(self, context: ToolContext) -> None:
        self.context = context

    def __call__(self, answer: str) -> str:
        """Judge whether an answer is truly complete and specific enough.
        This tool helps determine if we've actually solved the problem or if
        we could be more specific.
        
        Parameters:
            answer: The answer or solution to evaluate
        
        Returns:
            str: Analysis of the answer's completeness and specificity
        """
        try:
            response = ollama.chat(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a strict judge evaluating whether an answer is complete and specific.
                        Consider:
                        1. Does the answer address all aspects of the original question?
                        2. Are there any ambiguities or vague statements?
                        3. Could the answer be more specific or detailed?
                        4. Are there any assumptions that should be verified?
                        5. Is there missing context that would be helpful?
                        
                        Conclude with either:
                        "JUDGMENT: COMPLETE" - if the answer is thorough and specific
                        "JUDGMENT: INCOMPLETE" - if the answer needs more detail or clarification
                        """
                    },
                    *self.context.messages,
                    {
                        "role": "user",
                        "content": f"Judge this answer: \n\n{answer}\n\nIs this answer complete and specific enough?"
                    }
                ],
                stream=False,
                tools=_TOOLS,
            )
            # Add judgment to messages
            self.context.messages.append({"role": "assistant", "content": response.message.content})
            # Add any tool calls to the context
            self.context.tool_calls.extend(response.message.tool_calls or [])
            # Set force_stop based on judgment
            self.context.force_stop = "JUDGMENT: COMPLETE" in response.message.content
            return response.message.content
        except Exception as e:
            return f"Error from judge: {str(e)}"


@tool
class HumanJudge:
    name = "human_judge"
    def __init__(self, context: ToolContext) -> None:
        self.context = context

    def __call__(self, question: str) -> str:
        """Ask the human user if they are satisfied with the result.
        Collects feedback and additional instructions if needed.
        
        Parameters:
            question: The question or task being evaluated
        
        Returns:
            str: Human feedback and any additional instructions
        """
        print("\n=== Human Judgment Required ===")
        print("-" * 40)
        print("Question:", question)
        print("-" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Accept - The answer is complete and satisfactory")
            print("2. Reject - The answer needs more work")
            
            choice = input("\nYour choice (1-2): ").strip()
            
            if choice == "1":
                return "HUMAN JUDGMENT: COMPLETE\nUser accepted the answer as complete and satisfactory."
            elif choice == "2":
                feedback = input("\nEnter your feedback or instructions:\n").strip()
                msg = f"HUMAN JUDGMENT: INCOMPLETE\nUser rejected the answer as incomplete or unsatisfactory.\nFeedback: {feedback}"
                self.context.messages.append({"role": "user", "content": msg})
                return msg
            else:
                print("Invalid choice. Please try again.")


app = typer.Typer()

def print_and_buffer(output_buffer: list[str], *lines: str) -> None:
    """Print lines immediately and add them to the buffer."""
    for line in lines:
        print(line)
        output_buffer.append(line)
    print()  # Extra newline for readability
    output_buffer.append("")


ENCODING = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding

def count_tokens(messages: list[dict]) -> int:
    """Count the number of tokens in a list of messages."""
    return sum(len(ENCODING.encode(str(m))) for m in messages)


@app.command(name="")
def run(
    prompt: Annotated[str | None, typer.Argument(help="The prompt to send to the model. Defaults to stdin.")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="The primary model to use.")] = DEFAULT_MODEL,
    vim: Annotated[bool, typer.Option("--vim", help="Open the output in vim.")] = False,
) -> None:
    """Run the model with the given prompt."""
    if prompt is None and sys.stdin.isatty():
        typer.echo("No prompt provided and stdin is not a tty, exiting.")
        return
    
    messages = [{"role": "user", "content": f"{EXTRA_SYSTEM_PROMPT}\n\n{prompt or sys.stdin.read()}"}]
    start = time.time()
    output_buffer = []
    model_overrides = []
    
    def execute_tool_calls(tool_calls):
        """Execute a list of tool calls and add results to messages."""
        if not tool_calls:
            return False
            
        print_and_buffer(output_buffer, "=== Tool Calls ===")
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            
            print_and_buffer(output_buffer, f"Tool: {fn_name}")
            
            try:
                tool_fn = next(t for t in _TOOLS if t.__name__ == fn_name)
                if getattr(tool_fn, '_is_class_tool', False):
                    context = ToolContext(messages, tool_calls)
                    instance = tool_fn._tool_class(context)
                    result = instance(**fn_args)
                    if context.force_stop:
                        return True
                else:
                    result = tool_fn(**fn_args)
            except Exception as e:
                result = f"ERROR: Tool {fn_name} failed with exception: {str(e)}"
                print_and_buffer(output_buffer, "Error:", result)
            else:
                print_and_buffer(output_buffer, "Result:", str(result))
            
            tool_result = ToolCallResult(tool_call=tool_call.function, result=result)
            messages.append({"role": "user", "content": tool_result.to_prompt()})
        return False

    while True:
        response = ollama.chat(
            model=(len(model_overrides) and model_overrides.pop() or model),
            messages=messages,
            stream=False,
            tools=_TOOLS,
        )
        
        print_and_buffer(output_buffer,
            "=== Response ===",
            response.message.content,
            f"[Tokens: {count_tokens(messages)}]"
        )
        
        # Maintain a single list of tool calls
        tool_calls = list(response.message.tool_calls or [])
        context = ToolContext(messages, tool_calls)
        
        if "RESULT: INCONCLUSIVE" in response.message.content:
            print_and_buffer(output_buffer, "=== Consulting Project Planner ===")
            try:
                planner = ProjectPlanner(context)
                result = planner("What do I do next?")
                print_and_buffer(output_buffer,
                    "Project Planner suggests:",
                    result
                )
            except Exception as e:
                print_and_buffer(output_buffer,
                    "Error consulting project planner:",
                    str(e)
                )
        
        # Execute all collected tool calls and check if we should stop
        if execute_tool_calls(tool_calls):
            break
        
        # If we might be done, consult both judges
        if not tool_calls and "RESULT: INCONCLUSIVE" not in response.message.content:
            # First consult the AI judge
            print_and_buffer(output_buffer, "=== Consulting Judge ===")
            try:
                judge = Judge(context)
                result = judge(answer=response.message.content)
                print_and_buffer(output_buffer,
                    "Judge's verdict:",
                    result
                )
                # If AI judge approves, ask the human
                if context.force_stop:
                    human_judge = HumanJudge(context)
                    result = human_judge(question="Is this answer satisfactory?")
                    print_and_buffer(output_buffer,
                        "Human's verdict:",
                        result
                    )
                    # Only stop if both judges approve
                    context.force_stop = "HUMAN JUDGMENT: COMPLETE" in result
                if context.force_stop:
                    break
            except Exception as e:
                print_and_buffer(output_buffer,
                    "Error consulting judges:",
                    str(e)
                )
    
    end = time.time()
    print_and_buffer(output_buffer,
        "=== Summary ===",
        f"✅ {_num_tool_calls} tool calls, {end - start:.2f}s"
    )
    
    if vim:
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
