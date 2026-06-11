import pathlib

def find_project_root() -> pathlib.Path:
    """Walk up from cwd to the dir containing pyproject.toml (fallback: cwd)."""
    project_root = pathlib.Path.cwd()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            return project_root
        project_root = project_root.parent
    return pathlib.Path.cwd()