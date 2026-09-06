import ast
import json
from pathlib import Path

def build_repo_map(repo_root: Path, repo_id: str) -> dict:
    file_tree = []
    symbol_index = {}
    entrypoints = []

    for py_file in repo_root.rglob("*.py"):
        if any(part in {".git", "venv", "__pycache__", "node_modules"} for part in py_file.parts):
            continue
        rel_path = str(py_file.relative_to(repo_root))
        file_tree.append(rel_path)

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        symbols = []
        is_entrypoint = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append({
                    "name": node.name,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                })
            elif isinstance(node, ast.If) and isinstance(getattr(node, "test", None), ast.Compare):
                if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                    is_entrypoint = True
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "app":
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "FastAPI":
                            is_entrypoint = True
                            
        if symbols:
            symbol_index[rel_path] = symbols
        if is_entrypoint:
            entrypoints.append(rel_path)

    dependencies = _parse_dependencies(repo_root)

    repo_map = {
        "repo_id": repo_id,
        "file_tree": sorted(file_tree),
        "symbol_index": symbol_index,
        "dependencies": dependencies,
        "entrypoints": entrypoints,
    }
    return repo_map


def _parse_dependencies(repo_root: Path) -> list[str]:
    deps = []
    req_file = repo_root / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line.split("==")[0].split(">=")[0].strip())
    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        # lightweight parse — swap for tomllib if you want full correctness
        for line in pyproject.read_text().splitlines():
            if "=" in line and not line.strip().startswith("["):
                deps.append(line.split("=")[0].strip().strip('"'))
    return deps


def save_repo_map(repo_map: dict, storage_dir: Path) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)
    out_path = storage_dir / f"{repo_map['repo_id']}.json"
    out_path.write_text(json.dumps(repo_map, indent=2))


def load_repo_map(repo_id: str, storage_dir: Path) -> dict | None:
    path = storage_dir / f"{repo_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())