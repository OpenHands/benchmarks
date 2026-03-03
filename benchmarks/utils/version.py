import os
import subprocess
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
    Get the current git sha from the SDK submodule.
    """
    return _get_submodule_sha(PROJECT_ROOT / "vendor" / "software-agent-sdk")


SDK_SHA = get_sdk_sha()
SDK_SHORT_SHA = SDK_SHA[:7]

# Centralized image tag prefix used by all benchmark runners.
#
# Docker image tags follow the format: <prefix>-<custom_tag>-<target>
# e.g. "abc1234-sweb.eval.x86_64.django_1776_django-12155-binary"
#
# By default this is the SDK submodule short SHA. Set the IMAGE_TAG_PREFIX
# environment variable to override (e.g. when using pre-built images from
# a different SDK revision or a CI-provided tag).
IMAGE_TAG_PREFIX = os.getenv("IMAGE_TAG_PREFIX", SDK_SHORT_SHA)
