"""Base utilities for test report generation.

Provides git-aware report naming so each test run records the commit it ran
against. Test files use it via:

    if __name__ == "__main__":
        from .base import run_tests_with_report
        sys.exit(run_tests_with_report(__file__, 'analyzer'))
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_git_info() -> Tuple[Optional[str], bool, bool]:
    """Return (commit_sha8, is_dirty, git_available)."""
    try:
        commit_sha = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        is_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip())
        return commit_sha, is_dirty, True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None, False, False


def generate_report_name(test_name: str, commit_sha: Optional[str], is_dirty: bool, extension: str = "log") -> str:
    if commit_sha:
        dirty_suffix = "_dirty" if is_dirty else ""
        return f"report_{test_name}_{commit_sha}{dirty_suffix}.{extension}"
    return f"report_{test_name}.{extension}"


def run_tests_with_report(test_file: str, test_name: str) -> int:
    """Run pytest on `test_file` and write a git-aware report under tests/reports/."""
    commit_sha, is_dirty, git_available = get_git_info()
    report_name = generate_report_name(test_name, commit_sha, is_dirty)

    reports_dir = Path(test_file).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / report_name

    if git_available:
        print(f"Git commit: {commit_sha}")
        print(f"Working directory: {'dirty (uncommitted changes)' if is_dirty else 'clean'}")
    else:
        print("Git not available")
    print(f"Report will be saved to: {report_path}")
    print("=" * 70)

    result = subprocess.run(
        ["pytest", test_file, "-v", "--tb=short"],
        capture_output=True, text=True,
    )

    header = (
        f"{'=' * 70}\nTest Report\n{'=' * 70}\n"
        f"Test: {test_name}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Git Commit: {commit_sha if git_available else 'N/A'}\n"
        f"Working Dir: {'dirty (uncommitted changes)' if is_dirty else 'clean'}\n"
        f"Exit Code: {result.returncode}\n{'=' * 70}\n\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n" + result.stderr)

    print(result.stdout)
    if result.stderr:
        print("\n=== STDERR ===\n" + result.stderr)
    print("=" * 70)
    print("All tests passed" if result.returncode == 0 else f"Some tests failed (exit {result.returncode})")
    print(f"Report saved: {report_path}")
    return result.returncode


__all__ = ["get_git_info", "generate_report_name", "run_tests_with_report"]
