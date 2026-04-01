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
    os.getenv("IMAGE_TAG_PREFIX") or _deprecated_sdk_short_sha or SDK_SHORT_SHA
)


def get_phased_image_tag_prefix() -> str:
    """Return the image tag prefix for phased-build benchmarks (swebench, swebenchmultimodal, swtbench).

    Phased-build assembly images include the Dockerfile content hash in
    their tags so that Dockerfile changes invalidate cached assemblies.
    The tag format is: ``{sdk_short_sha}-{content_hash}-{custom_tag}-{target}``.

    Benchmarks on the legacy build path (gaia, commit0, etc.) should
    continue to use :data:`IMAGE_TAG_PREFIX` which does NOT include the
    content hash.
    """
    from benchmarks.swebench.build_base_images import dockerfile_content_hash

    # Only IMAGE_TAG_PREFIX (explicit override) bypasses the content hash.
    # The deprecated SDK_SHORT_SHA env var must NOT short-circuit here,
    # because run_eval.sh sets it and we need the content hash included.
    return (
        os.getenv("IMAGE_TAG_PREFIX") or f"{SDK_SHORT_SHA}-{dockerfile_content_hash()}"
    )
