import os


def get_relative_path(
    local_repo_path: str,
    absolute_path: str
) -> str:

    return os.path.relpath(
        absolute_path,
        local_repo_path
    )
import os


def get_relative_path(repo_path: str, absolute_path: str) -> str:
    """
    Converts absolute file path into repo-relative path.
    """

    return os.path.relpath(absolute_path, repo_path)


def read_file(repo_path: str, relative_path: str) -> str:
    """
    Reads a file from the cloned repository.
    """

    full_path = os.path.join(repo_path, relative_path)

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
