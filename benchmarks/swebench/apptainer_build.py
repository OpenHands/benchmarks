"""Local Apptainer builds for SWE-bench agent-server images."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from benchmarks.swebench import constants
from benchmarks.swebench.build_base_images import dockerfile_content_hash
from benchmarks.utils.build_utils import BuildOutput, _get_sdk_submodule_info
from openhands.sdk import get_logger


logger = get_logger(__name__)

DEFAULT_APPTAINER_BUILD_ROOT = (
    Path.home() / ".cache" / "openhands" / "swebench-apptainer-agent-images"
)
SUPPORTED_APPTAINER_TARGETS = {constants.BUILD_TARGET_SOURCE_MINIMAL}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sdk_root() -> Path:
    return _repo_root() / "vendor" / "software-agent-sdk"


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _build_root() -> Path:
    return Path(
        os.getenv("OPENHANDS_APPTAINER_BUILD_ROOT", str(DEFAULT_APPTAINER_BUILD_ROOT))
    ).expanduser()


def _force_build_enabled() -> bool:
    return os.getenv("OPENHANDS_APPTAINER_FORCE_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
    }


def apptainer_agent_image_path(
    custom_tag: str,
    target: constants.TargetType = constants.DEFAULT_BUILD_TARGET,
) -> Path:
    """Return the local Apptainer SIF path for a SWE-bench agent image."""
    _, git_sha, _ = _get_sdk_submodule_info()
    sdk_short_sha = git_sha[:7] if git_sha != "unknown" else "unknown"
    content_hash = dockerfile_content_hash()
    name = _sanitize_filename(f"{sdk_short_sha}-{content_hash}-{custom_tag}-{target}")
    return _build_root() / f"{name}.sif"


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _package_install_script() -> str:
    """Return package setup shell matching the minimal Docker target."""
    return r"""
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
    apt-get -o Acquire::Retries=5 update
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        bash ca-certificates curl wget sudo apt-utils git jq tmux tar \
        build-essential coreutils util-linux procps findutils grep sed \
        apt-transport-https gnupg lsb-release xz-utils
    rm -rf /var/lib/apt/lists/*
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache \
        bash ca-certificates curl wget sudo git jq tmux tar build-base \
        coreutils util-linux procps findutils grep sed gnupg shadow xz
elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps-ng findutils grep sed shadow-utils \
        gnupg2 xz
    microdnf clean all
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps-ng findutils grep sed shadow-utils \
        gnupg2 xz
    dnf clean all
elif command -v yum >/dev/null 2>&1; then
    yum install -y \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps-ng findutils grep sed shadow-utils \
        gnupg2 xz
    yum clean all
elif command -v zypper >/dev/null 2>&1; then
    zypper --non-interactive install --no-recommends \
        bash ca-certificates curl wget sudo git jq tmux tar make gcc gcc-c++ \
        coreutils util-linux procps findutils grep sed shadow gpg2 xz
    zypper clean --all
else
    echo "Unsupported base image: no known package manager found" >&2
    exit 1
fi
"""


def _wrap_swebench_deps_script() -> str:
    """Return optional Sphinx dependency wrapper shell."""
    return r"""
if command -v conda >/dev/null 2>&1; then
    conda run -n testbed pip install --no-deps --force-reinstall 'docutils<0.21' 'roman' \
        || (source /opt/miniconda3/bin/activate testbed && pip install --no-deps --force-reinstall 'docutils<0.21' 'roman')
elif [ -x /opt/miniconda3/bin/conda ]; then
    /opt/miniconda3/bin/conda run -n testbed pip install --no-deps --force-reinstall 'docutils<0.21' 'roman' \
        || (source /opt/miniconda3/bin/activate testbed && pip install --no-deps --force-reinstall 'docutils<0.21' 'roman')
fi
if command -v pip >/dev/null 2>&1; then
    pip install --no-deps --force-reinstall 'docutils<0.21' 'roman'
fi
"""


def _definition_file_content(
    base_image: str,
    git_sha: str,
    git_ref: str,
    wrap_swebench_deps: bool,
    uv_path: Path,
    uvx_path: Path | None,
) -> str:
    sdk_root = _sdk_root()
    wrap_script = _wrap_swebench_deps_script() if wrap_swebench_deps else ""
    uvx_files = f"    {uvx_path} /usr/local/bin/uvx\n" if uvx_path else ""
    uv_concurrent_downloads = os.getenv(
        "OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS", "4"
    )
    uv_concurrent_builds = os.getenv("OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS", "1")
    uv_concurrent_installs = os.getenv(
        "OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS", "1"
    )
    return f"""Bootstrap: docker
From: {base_image}

%files
    {uv_path} /usr/local/bin/uv
{uvx_files}\
    {sdk_root / "pyproject.toml"} /agent-server/pyproject.toml
    {sdk_root / "uv.lock"} /agent-server/uv.lock
    {sdk_root / "README.md"} /agent-server/README.md
    {sdk_root / "LICENSE"} /agent-server/LICENSE
    {sdk_root / "openhands-sdk"} /agent-server/openhands-sdk
    {sdk_root / "openhands-tools"} /agent-server/openhands-tools
    {sdk_root / "openhands-workspace"} /agent-server/openhands-workspace
    {sdk_root / "openhands-agent-server"} /agent-server/openhands-agent-server

%post
    set -eux
    {_package_install_script()}

    USERNAME=openhands
    UID=10001
    GID=10001
    grep -Eq "^[^:]*:[^:]*:${{GID}}:" /etc/group || groupadd -g "${{GID}}" "${{USERNAME}}"
    grep -Eq "^${{USERNAME}}:" /etc/passwd || useradd -m -u "${{UID}}" -g "${{GID}}" -s /bin/bash "${{USERNAME}}"
    usermod -aG sudo "${{USERNAME}}" 2>/dev/null || true
    echo "${{USERNAME}} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    mkdir -p /workspace/project /agent-server/uv-managed-python
    chown -R "${{USERNAME}}:${{USERNAME}}" /workspace /agent-server

    chmod 0755 /usr/local/bin/uv
    if [ -e /usr/local/bin/uvx ]; then chmod 0755 /usr/local/bin/uvx; fi

    su "${{USERNAME}}" -s /bin/bash -c 'cd /agent-server && \\
        export HOME=/home/openhands && \\
        export UV_CONCURRENT_DOWNLOADS={uv_concurrent_downloads} && \\
        export UV_CONCURRENT_BUILDS={uv_concurrent_builds} && \\
        export UV_CONCURRENT_INSTALLS={uv_concurrent_installs} && \\
        export UV_PROJECT_ENVIRONMENT=/agent-server/.venv && \\
        export UV_PYTHON_INSTALL_DIR=/agent-server/uv-managed-python && \\
        uv python install 3.13 && \\
        uv venv --python-preference only-managed --python 3.13 .venv && \\
        uv sync --frozen --no-editable --managed-python --extra boto3 && \\
        uv pip install --python /agent-server/.venv/bin/python "transformers>=4.56.0,<5" && \\
        readlink -f .venv/bin/python | grep -q "^/agent-server/uv-managed-python/"'

    {wrap_script}

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export OH_ENABLE_VNC=false
    export LOG_JSON=true
    export OPENHANDS_BUILD_GIT_SHA={git_sha}
    export OPENHANDS_BUILD_GIT_REF={git_ref}

%runscript
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export OH_ENABLE_VNC=false
    export LOG_JSON=true
    export OPENHANDS_BUILD_GIT_SHA={git_sha}
    export OPENHANDS_BUILD_GIT_REF={git_ref}
    exec /agent-server/.venv/bin/python -m openhands.agent_server "$@"
"""


def build_apptainer_agent_image(
    base_image: str,
    custom_tag: str,
    target: constants.TargetType = constants.DEFAULT_BUILD_TARGET,
    wrap_swebench_deps: bool = False,
) -> BuildOutput:
    """Build a local Apptainer agent-server SIF from a SWE-bench base image."""
    if target not in SUPPORTED_APPTAINER_TARGETS:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=(
                f"Apptainer local builds currently support "
                f"{sorted(SUPPORTED_APPTAINER_TARGETS)}, got {target!r}"
            ),
        )

    if shutil.which("apptainer") is None:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error="Apptainer is not available on PATH",
        )
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error="uv is not available on PATH",
        )
    uvx_bin = shutil.which("uvx")

    image_path = apptainer_agent_image_path(custom_tag, target)
    if image_path.exists() and not _force_build_enabled():
        logger.info("Using existing Apptainer agent SIF %s", image_path)
        return BuildOutput(base_image=base_image, tags=[str(image_path)], error=None)

    build_root = _build_root()
    log_dir = build_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    tmp_image = image_path.with_suffix(".tmp.sif")
    _remove_path(tmp_image)
    _remove_path(image_path)

    git_ref, git_sha, _ = _get_sdk_submodule_info()
    definition = build_root / f"{image_path.name}.def"
    definition.write_text(
        _definition_file_content(
            base_image=base_image,
            git_sha=git_sha,
            git_ref=git_ref,
            wrap_swebench_deps=wrap_swebench_deps,
            uv_path=Path(uv_bin).resolve(),
            uvx_path=Path(uvx_bin).resolve() if uvx_bin else None,
        )
    )

    log_path = log_dir / f"{image_path.name}.log"
    cmd = ["apptainer", "build", str(tmp_image), str(definition)]
    logger.info("Building Apptainer agent SIF: %s", " ".join(cmd))
    env = os.environ.copy()
    if "APPTAINER_CACHEDIR" not in env:
        env["APPTAINER_CACHEDIR"] = str(build_root / "cache")
    for key in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        if env.get(key):
            Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        _remove_path(tmp_image)
        return BuildOutput(
            base_image=base_image,
            tags=[],
            error=f"Apptainer build failed with exit code {proc.returncode}",
            log_path=str(log_path),
        )

    tmp_image.rename(image_path)
    logger.info("Built Apptainer agent SIF %s", image_path)
    return BuildOutput(
        base_image=base_image,
        tags=[str(image_path)],
        error=None,
        log_path=str(log_path),
    )


def ensure_apptainer_agent_image(
    base_image: str,
    custom_tag: str,
    target: constants.TargetType = constants.DEFAULT_BUILD_TARGET,
    wrap_swebench_deps: bool = False,
) -> Path:
    """Build or reuse a local Apptainer agent-server SIF."""
    output = build_apptainer_agent_image(
        base_image=base_image,
        custom_tag=custom_tag,
        target=target,
        wrap_swebench_deps=wrap_swebench_deps,
    )
    logger.info("Apptainer image build output: %s", output)
    if output.error is not None:
        raise RuntimeError(f"Apptainer image build failed: {output.error}")
    if not output.tags:
        raise RuntimeError("Apptainer image build produced no image path")
    return Path(output.tags[0])
