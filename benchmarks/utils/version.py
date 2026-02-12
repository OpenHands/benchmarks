import os
import subprocess
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_submodule_sha(submodule_path: Path) -> str:
    result = subprocess.run(
        ["git", "submodule", "status", str(submodule_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    sha = result.stdout.strip().split()[0].lstrip("+-")
    return sha


def get_sdk_sha() -> str:
    """
    Get the SDK SHA from git submodule, falling back to "unknown".
    """
    try:
        return _get_submodule_sha(PROJECT_ROOT / "vendor" / "software-agent-sdk")
    except subprocess.CalledProcessError:
        warnings.warn(
            "Could not get SDK SHA from git submodule. Using 'unknown' as fallback. "
        )
        return "unknown"


SDK_SHA = get_sdk_sha()
SDK_SHORT_SHA = SDK_SHA[:7]

# This is used as the first part of the image tag: <prefix>-<custom_tag>-<target>
IMAGE_TAG_PREFIX = os.getenv("IMAGE_TAG_PREFIX", SDK_SHORT_SHA)
