import json
from langchain_core.tools import tool
import subprocess, pathlib

WORKSPACE = pathlib.Path("workspace").resolve()

def safe_path(p: str) -> pathlib.Path:
    resolved = (WORKSPACE / p).resolve()
    assert resolved.is_relative_to(WORKSPACE), "Path escape attempt"
    return resolved

@tool
def read_file(path: str) -> str:
    """Read the full contents of a file from the workspace.

    Args:
        path: Relative path to the file from the workspace root (e.g. 'src/main.py').
              Must not escape the workspace via '..'.
    """
    try:
        return safe_path(path).read_text()
    except Exception as e:
        return str(e)

@tool
def write_file(path: str, content: str) -> str:
    """Write (or overwrite) a file in the workspace with the given content.
    Prefer str_replace for targeted edits to avoid rewriting unchanged content.
    Creates parent directories automatically if they don't exist.

    Args:
        path: Relative path to the file from the workspace root (e.g. 'src/utils.py').
              Must not escape the workspace via '..'.
        content: Full text content to write to the file. Overwrites existing content entirely.
    """
    if isinstance(content, dict):
        content = json.dumps(content)
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} bytes to {path}"

@tool
def str_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace a unique string in a file with new content. Preferred over write_file
    for targeted edits — cheaper on tokens and less error-prone.
    Fails if old_str appears 0 or 2+ times (must be unique).

    Args:
        path: Relative path to the file from the workspace root (e.g. 'src/main.py').
        old_str: The exact string to find and replace. Must appear exactly once in the file.
                 Include enough surrounding context (e.g. the full function signature)
                 to make it unique.
        new_str: The string to replace old_str with. Can be empty to delete old_str.
    """
    p = safe_path(path)
    text = p.read_text()
    if text.count(old_str) != 1:
        return f"Error: found {text.count(old_str)} occurrences, need exactly 1"
    p.write_text(text.replace(old_str, new_str, 1))
    return "Done"

@tool
def bash(command: str, timeout: int = 900) -> str:
    """Execute a shell command in the workspace directory and return stdout + stderr.
    Use for running tests, installing packages, listing files, or any system operation.
    Output is truncated to 8000 characters if large.

    Args:
        command: Shell command to run (e.g. 'pytest tests/', 'pip install requests',
                 'find . -name "*.py"'). Runs with workspace as the working directory.
        timeout: Maximum seconds to wait before killing the process. Defaults to 30.
                 Increase for long-running operations like installs or test suites.
    """
    if "cd " in command:
        return "Error: changing directories is not allowed"
    if "wget " in command:
        return "Error: downloading from internet is not allowed"
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=WORKSPACE,
        stdin=subprocess.DEVNULL
    )
    out = (result.stdout + result.stderr)
    return out or "(no output)"
