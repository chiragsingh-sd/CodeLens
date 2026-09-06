import ast
import hashlib
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
            "language": "python",
            "truncated": False,
            "original_char_count": 120,
            "content_hash": "..."
        }
    }
    """
    relative_path = get_relative_path(local_repo_path, absolute_path)

    try:
        with open(
            absolute_path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:
            source = f.read()
    except Exception:
        return []

    if not source.strip():
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        line_count = len(source.splitlines())
        original_char_count = len(source)
        content_hash = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()[:16]

        truncated = len(source) > 3000

        text = (
            source[:3000] + "\n# ... (truncated)"
            if truncated
            else source
        )

        return [{
            "text": text,
            "metadata": {
                "file_path": relative_path,
                "symbol_name": os.path.basename(absolute_path),
                "chunk_type": "module",
                "start_line": 1,
                "end_line": line_count,
                "language": "python",
                "truncated": truncated,
                "original_char_count": original_char_count,
                "content_hash": content_hash
            }
        }]

    source_lines = source.splitlines()
    chunks = []

    nodes_to_chunk = []

    for node in tree.body:
        is_function = isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        is_class = isinstance(node, ast.ClassDef)

        if is_function or is_class:
            nodes_to_chunk.append((node, node.name, "function" if is_function else "class"))
            
        if is_class:
            for inner_node in node.body:
                if isinstance(inner_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nodes_to_chunk.append((inner_node, f"{node.name}.{inner_node.name}", "method"))

    for node, symbol_name, chunk_type in nodes_to_chunk:

        start = node.lineno - 1
        end = node.end_lineno

        chunk_lines = source_lines[start:end]
        chunk_text = "\n".join(chunk_lines)

        if len(chunk_lines) < 3:
            continue

        original_char_count = len(chunk_text)

        content_hash = hashlib.sha256(
            chunk_text.encode("utf-8")
        ).hexdigest()[:16]

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "file_path": relative_path,
                "symbol_name": symbol_name,
                "chunk_type": chunk_type,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "language": "python",
                "truncated": False,
                "original_char_count": original_char_count,
                "content_hash": content_hash
            }
        })

    if not chunks:
        original_char_count = len(source)

        content_hash = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()[:16]

        truncated = len(source) > 2000

        text = (
            source[:2000] + "\n# ... (truncated)"
            if truncated
            else source
        )

        chunks.append({
            "text": text,
            "metadata": {
                "file_path": relative_path,
                "symbol_name": os.path.basename(absolute_path),
                "chunk_type": "module",
                "start_line": 1,
                "end_line": len(source_lines),
                "language": "python",
                "truncated": truncated,
                "original_char_count": original_char_count,
                "content_hash": content_hash
            }
        })

    return chunks


def chunk_repo(
    python_files: list[str],
    local_repo_path: str
) -> list[dict]:
    """
    Chunk every Python file in the repo.
    Returns all chunks from all files combined.
    """
    all_chunks = []

    for file_path in python_files:
        file_chunks = chunk_python_file(
            file_path,
            local_repo_path
        )
        all_chunks.extend(file_chunks)

    return all_chunks   