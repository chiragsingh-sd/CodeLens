import os
import subprocess
from backend.tools.file_tool import list_structure


def find_definition(local_repo_path: str, symbol_name: str) -> str:
    """
    Find where a function or class is defined in the repo.
    
    First tries grep (fast, Unix/Mac/Linux).
    Falls back to pure Python search (works on Windows too).
    """
    try:
        result = subprocess.run(
            ["grep", "-rn",
             f"def {symbol_name}\\|class {symbol_name}",
             local_repo_path, "--include=*.py"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            output = result.stdout.replace(local_repo_path + "/", "")
            return f"Definition(s) found:\n{output[:2000]}"
        return f"No definition found for '{symbol_name}' in Python files."

    except FileNotFoundError:
        # Use a Python fallback when grep is unavailable on Windows.
        return _find_definition_python(local_repo_path, symbol_name)
    except Exception as e:
        return f"Error searching: {e}"


def _find_definition_python(local_repo_path: str, symbol_name: str) -> str:
    """Pure Python fallback for find_definition."""
    results = []
    for root, dirs, files in os.walk(local_repo_path):
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d not in {"__pycache__", "venv"}]
        for file in files:
            if not file.endswith(".py"):
                continue
            full_path = os.path.join(root, file)
            rel_path  = os.path.relpath(full_path, local_repo_path)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if f"def {symbol_name}" in line or f"class {symbol_name}" in line:
                            results.append(f"{rel_path}:{lineno}: {line.strip()}")
            except Exception:
                continue

    if results:
        return "Definitions found:\n" + "\n".join(results[:20])
    return f"No definition found for '{symbol_name}'"
