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


def _get_dockerfile_content_hash() -> str:
    """Return the 7-char content hash of the SDK Dockerfile.

    This is the same hash used to tag eval-base images so that Dockerfile
    changes auto-invalidate cached images.  We include it in the image tag
    prefix so that the pull side constructs tags that match the build side.
    """
    from benchmarks.swebench.build_base_images import dockerfile_content_hash

    return dockerfile_content_hash()


# Centralized image tag prefix used by all benchmark runners.
#
# Docker image tags follow the format: <prefix>-<custom_tag>-<target>
# e.g. "acd5adc-245a238-sweb.eval.x86_64.django_1776_django-12155-source-minimal"
#
# The prefix includes both the SDK submodule short SHA and the Dockerfile
# content hash so that changes to either invalidate cached images.
# Set the IMAGE_TAG_PREFIX environment variable to override.
# Check for deprecated env var and warn users
_deprecated_sdk_short_sha = os.getenv("SDK_SHORT_SHA")
if _deprecated_sdk_short_sha is not None:
    import warnings

    warnings.warn(
        "SDK_SHORT_SHA environment variable is deprecated. Use IMAGE_TAG_PREFIX instead.",
        DeprecationWarning,
        stacklevel=2,
    )

IMAGE_TAG_PREFIX = (
    os.getenv("IMAGE_TAG_PREFIX")
    or _deprecated_sdk_short_sha
    or f"{SDK_SHORT_SHA}-{_get_dockerfile_content_hash()}"
)
