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
#
# Backward compatibility: SDK_SHORT_SHA env var is still honored as a
# fallback when IMAGE_TAG_PREFIX is not set, but is deprecated.
_image_tag_prefix_env = os.getenv("IMAGE_TAG_PREFIX")
_sdk_short_sha_env = os.getenv("SDK_SHORT_SHA")

if _image_tag_prefix_env is not None:
    IMAGE_TAG_PREFIX = _image_tag_prefix_env
elif _sdk_short_sha_env is not None:
    warnings.warn(
        "SDK_SHORT_SHA env var is deprecated for overriding image tags. "
        "Use IMAGE_TAG_PREFIX instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    IMAGE_TAG_PREFIX = _sdk_short_sha_env
else:
    IMAGE_TAG_PREFIX = SDK_SHORT_SHA
