
import os
import re
import shutil
import git
from backend.core.config import settings


def parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract owner and repo name from a GitHub URL.
    
    "https://github.com/biswaisop/Genos"  →  ("biswaisop", "Genos")
    "https://github.com/biswaisop/Genos/" →  ("biswaisop", "Genos")  [trailing slash]
    """
    match = re.search(r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', url.strip())
    if not match:
        raise ValueError(f"Not a valid GitHub URL: {url}")
    owner = match.group(1)
    name  = match.group(2)
    return owner, name


def clone_repo(repo_id: str, url: str) -> str:
    """
    Clone the GitHub repo to disk. Returns the local path.
    
    Raises an exception if the repo doesn't exist or is private.
    """
    local_path = os.path.join(settings.repo_storage_path, repo_id)

    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    os.makedirs(local_path, exist_ok=True)

    # History is irrelevant to code search, so keep clones shallow.
    git.Repo.clone_from(
        url,
        local_path,
        depth=1,
    )

    return local_path


def get_python_files(local_path: str, max_files: int) -> list[str]:
    """
    Walk the cloned repo and return paths of all .py files.
    Skips: test files, migrations, __pycache__, hidden folders.
    
    Returns a list of absolute file paths.
    """
    python_files = []

    for root, dirs, files in os.walk(local_path):

        dirs[:] = [
            d for d in dirs
            if d not in {'__pycache__', '.git', 'node_modules',
                         'venv', 'env', '.venv', 'migrations'}
            and not d.startswith('.')
        ]

        for file in files:
            if not file.endswith('.py'):
                continue
            if file.startswith('test_') or file.endswith('_test.py'):
                continue

            full_path = os.path.join(root, file)
            python_files.append(full_path)

            if len(python_files) >= max_files:
                return python_files

    return python_files
