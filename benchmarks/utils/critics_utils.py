"""
Utility functions for critics in SWE-Bench evaluation.
"""

from benchmarks.utils.models import EvalOutput


def _has_non_empty_git_patch(output: EvalOutput) -> bool:
    """
    Check if the git patch is non-empty.

    Args:
        output: The evaluation output to check

    Returns:
        True if the git patch is non-empty, False otherwise
    """
    git_patch = output.test_result.get("git_patch", "")
    return bool(git_patch and git_patch.strip())
