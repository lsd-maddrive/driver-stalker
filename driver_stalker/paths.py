"""Module for project paths.

Don`t use it if you build package and use it externally!
"""

from pathlib import Path


def get_repo_dpath() -> Path:
    """Get the repository root folder path."""
    current_path = Path.cwd()
    for parent in [current_path, *list(current_path.parents)]:
        if (parent / ".git").exists():
            return parent

    err_msg = "Reposity root dpath is not found."
    raise FileNotFoundError(err_msg)
