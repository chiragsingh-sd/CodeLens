import ast
import os
from backend.tools.file_tool import get_relative_path


def chunk_python_file(absolute_path: str, local_repo_path: str) -> list[dict]:
    """
    Parse a Python file using Python's built-in AST module.
    Returns a list of chunks, where each chunk is one complete function or class.
    
    Each chunk is a dict:
    {
        "text": "def verify_password(plain, hashed):\n    ...",
        "metadata": {
            "file_path": "backend/auth.py",
            "symbol_name": "verify_password",
            "chunk_type": "function",
            "start_line": 42,
            "end_line": 48,
            "language": "python"
        }
    }
    """
    relative_path = get_relative_path(local_repo_path, absolute_path)

    try:
        with open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception:
        return []   # skip unreadable files silently

    if not source.strip():
        return []   # skip empty files

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Preserve syntactically invalid files as searchable module-level context.
        return [{
            "text": source[:3000],
            "metadata": {
                "file_path": relative_path,
                "symbol_name": os.path.basename(absolute_path),
                "chunk_type": "module",
                "start_line": 1,
                "end_line": len(source.splitlines()),
                "language": "python"
            }
        }]

    source_lines = source.splitlines()
    chunks = []

    # Top-level symbols keep each retrieved chunk aligned with a file-level unit.
    for node in tree.body:

        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_class    = isinstance(node, ast.ClassDef)

        if not (is_function or is_class):
            continue    # skip everything else (imports, assignments, etc.)

        start = node.lineno - 1
        end   = node.end_lineno      # end_lineno is inclusive

        chunk_lines = source_lines[start:end]
        chunk_text  = "\n".join(chunk_lines)

        if len(chunk_lines) < 3:
            continue

        # Bound prompt/index size; large classes are represented by their prefix.
        if len(chunk_text) > 1500:
            chunk_text = chunk_text[:1500] + "\n# ... (truncated)"

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "file_path":   relative_path,
                "symbol_name": node.name,
                "chunk_type":  "function" if is_function else "class",
                "start_line":  node.lineno,
                "end_line":    node.end_lineno,
                "language":    "python"
            }
        })

    # Module-level scripts still need a searchable representation.
    if not chunks:
        chunks.append({
            "text": source[:2000],
            "metadata": {
                "file_path":   relative_path,
                "symbol_name": os.path.basename(absolute_path),
                "chunk_type":  "module",
                "start_line":  1,
                "end_line":    len(source_lines),
                "language":    "python"
            }
        })

    return chunks


def chunk_repo(python_files: list[str], local_repo_path: str) -> list[dict]:
    """
    Chunk every Python file in the repo.
    Returns all chunks from all files combined.
    """
    all_chunks = []
    for file_path in python_files:
        file_chunks = chunk_python_file(file_path, local_repo_path)
        all_chunks.extend(file_chunks)
    return all_chunks
