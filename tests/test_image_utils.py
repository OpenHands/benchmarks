"""Tests for image_utils and build_utils helper functions.

Tests cover local_image_exists(), create_docker_workspace(), and ensure_local_image()
which centralize Docker image detection and build logic across all benchmarks.
"""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.utils.build_utils import BuildOutput


class TestLocalImageExists:
    """Tests for local_image_exists()."""

    @patch("benchmarks.utils.image_utils.subprocess.run")
    def test_image_exists(self, mock_run):
        from benchmarks.utils.image_utils import local_image_exists

        mock_run.return_value = MagicMock(returncode=0)
        assert local_image_exists("myimage:latest") is True
        mock_run.assert_called_once_with(
            ["docker", "image", "inspect", "myimage:latest"],
            capture_output=True,
            check=False,
            timeout=5,
        )

    @patch("benchmarks.utils.image_utils.subprocess.run")
    def test_image_not_found(self, mock_run):
        from benchmarks.utils.image_utils import local_image_exists

        mock_run.return_value = MagicMock(returncode=1)
        assert local_image_exists("noimage:latest") is False

    @patch("benchmarks.utils.image_utils.subprocess.run")
    def test_timeout_returns_false(self, mock_run):
        from benchmarks.utils.image_utils import local_image_exists

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=5)
        assert local_image_exists("myimage:latest") is False

    @patch("benchmarks.utils.image_utils.subprocess.run")
    def test_docker_not_installed_returns_false(self, mock_run):
        from benchmarks.utils.image_utils import local_image_exists

        mock_run.side_effect = FileNotFoundError("docker not found")
        assert local_image_exists("myimage:latest") is False


class TestCreateDockerWorkspace:
    """Tests for create_docker_workspace().

    These tests mock the Docker daemon interaction (local_image_exists) and
    workspace constructors (which connect to Docker), but verify the actual
    branching logic and argument forwarding.
    """

    @patch("benchmarks.utils.image_utils.local_image_exists", return_value=True)
    def test_returns_docker_workspace_when_image_exists(self, _mock_exists):
        from benchmarks.utils.image_utils import create_docker_workspace
        from openhands.workspace import DockerWorkspace

        with patch("openhands.workspace.DockerWorkspace", wraps=DockerWorkspace) as spy:
            # wraps=DockerWorkspace would call the real constructor which needs Docker,
            # so we set a return_value to avoid that while still checking isinstance
            sentinel = MagicMock(spec=DockerWorkspace)
            spy.return_value = sentinel
            ws = create_docker_workspace(
                agent_server_image="server:v1",
                base_image="base:latest",
                build_target="binary",
            )
            spy.assert_called_once_with(
                server_image="server:v1",
                working_dir="/workspace",
                forward_env=[],
            )
            assert ws is sentinel

    @patch("benchmarks.utils.image_utils.local_image_exists", return_value=False)
    def test_returns_docker_dev_workspace_when_image_missing(self, _mock_exists):
        from benchmarks.utils.image_utils import create_docker_workspace
        from openhands.workspace import DockerDevWorkspace

        sentinel = MagicMock(spec=DockerDevWorkspace)
        with patch(
            "openhands.workspace.DockerDevWorkspace", return_value=sentinel
        ) as spy:
            ws = create_docker_workspace(
                agent_server_image="server:v1",
                base_image="base:latest",
                build_target="source-minimal",
                forward_env=["FOO"],
            )
            spy.assert_called_once_with(
                base_image="base:latest",
                working_dir="/workspace",
                target="source-minimal",
                forward_env=["FOO"],
            )
            assert ws is sentinel

    @patch.dict(os.environ, {"FORCE_BUILD": "1"})
    @patch("benchmarks.utils.image_utils.local_image_exists", return_value=True)
    def test_force_build_skips_detection(self, mock_exists):
        from benchmarks.utils.image_utils import create_docker_workspace
        from openhands.workspace import DockerDevWorkspace

        sentinel = MagicMock(spec=DockerDevWorkspace)
        with patch("openhands.workspace.DockerDevWorkspace", return_value=sentinel):
            ws = create_docker_workspace(
                agent_server_image="server:v1",
                base_image="base:latest",
                build_target="binary",
            )
            # Should build even though image exists locally
            assert ws is sentinel
            # local_image_exists should NOT have been called when FORCE_BUILD=1
            mock_exists.assert_not_called()

    @patch("benchmarks.utils.image_utils.local_image_exists", return_value=True)
    def test_custom_working_dir_and_forward_env(self, _mock_exists):
        """Verify custom parameters are forwarded correctly."""
        from benchmarks.utils.image_utils import create_docker_workspace

        with patch("openhands.workspace.DockerWorkspace") as MockDW:
            create_docker_workspace(
                agent_server_image="server:v1",
                base_image="base:latest",
                build_target="binary",
                working_dir="/custom",
                forward_env=["API_KEY", "TOKEN"],
            )
            MockDW.assert_called_once_with(
                server_image="server:v1",
                working_dir="/custom",
                forward_env=["API_KEY", "TOKEN"],
            )


class TestEnsureLocalImage:
    """Tests for ensure_local_image().

    Uses real BuildOutput objects (not mocks) so validation logic in
    ensure_local_image is exercised against actual data structures.
    """

    @patch("benchmarks.utils.build_utils.local_image_exists", return_value=True)
    @patch("benchmarks.utils.build_utils.build_image")
    def test_returns_false_when_image_exists(self, mock_build, _mock_exists):
        from benchmarks.utils.build_utils import ensure_local_image

        result = ensure_local_image(
            agent_server_image="server:v1",
            base_image="base:latest",
            custom_tag="mytag",
        )
        assert result is False
        mock_build.assert_not_called()

    @patch("benchmarks.utils.build_utils.local_image_exists", return_value=False)
    @patch("benchmarks.utils.build_utils.build_image")
    def test_returns_true_when_build_occurs(self, mock_build, _mock_exists):
        from benchmarks.utils.build_utils import ensure_local_image

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=["server:v1"],
            error=None,
        )
        result = ensure_local_image(
            agent_server_image="server:v1",
            base_image="base:latest",
            custom_tag="mytag",
        )
        assert result is True
        mock_build.assert_called_once()

    @patch("benchmarks.utils.build_utils.local_image_exists", return_value=False)
    @patch("benchmarks.utils.build_utils.build_image")
    def test_raises_on_build_failure(self, mock_build, _mock_exists):
        from benchmarks.utils.build_utils import ensure_local_image

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=[],
            error="build exploded",
        )
        with pytest.raises(RuntimeError, match="Image build failed"):
            ensure_local_image(
                agent_server_image="server:v1",
                base_image="base:latest",
                custom_tag="mytag",
            )

    @patch("benchmarks.utils.build_utils.local_image_exists", return_value=False)
    @patch("benchmarks.utils.build_utils.build_image")
    def test_raises_on_tag_mismatch(self, mock_build, _mock_exists):
        from benchmarks.utils.build_utils import ensure_local_image

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=["server:wrong-tag"],
            error=None,
        )
        with pytest.raises(RuntimeError, match="do not include expected tag"):
            ensure_local_image(
                agent_server_image="server:v1",
                base_image="base:latest",
                custom_tag="mytag",
            )

    @patch.dict(os.environ, {"FORCE_BUILD": "1"})
    @patch("benchmarks.utils.build_utils.local_image_exists", return_value=True)
    @patch("benchmarks.utils.build_utils.build_image")
    def test_force_build_skips_detection(self, mock_build, mock_exists):
        from benchmarks.utils.build_utils import ensure_local_image

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=["server:v1"],
            error=None,
        )
        result = ensure_local_image(
            agent_server_image="server:v1",
            base_image="base:latest",
            custom_tag="mytag",
        )
        assert result is True
        mock_build.assert_called_once()
        # local_image_exists should NOT have been called when FORCE_BUILD=1
        mock_exists.assert_not_called()

    @patch("benchmarks.utils.build_utils.local_image_exists", return_value=False)
    @patch("benchmarks.utils.build_utils.build_image")
    def test_passes_target_to_build_image(self, mock_build, _mock_exists):
        """Verify the target parameter flows through to build_image."""
        from benchmarks.utils.build_utils import ensure_local_image

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=["server:v1"],
            error=None,
        )
        ensure_local_image(
            agent_server_image="server:v1",
            base_image="base:latest",
            custom_tag="mytag",
            target="binary",
        )
        _, kwargs = mock_build.call_args
        assert kwargs["target"] == "binary"
        assert kwargs["push"] is False


class TestCachedSdistReuse:
    def test_is_sdist_build_command_matches_expected_shape(self):
        from benchmarks.utils.build_utils import _is_sdist_build_command

        assert _is_sdist_build_command(
            ["uv", "build", "--sdist", "--out-dir", "/tmp/out"]
        )
        assert not _is_sdist_build_command(["uv", "build", "--wheel"])
        assert not _is_sdist_build_command(["docker", "buildx", "build"])

    def test_patch_sdk_sdist_build_reuses_cached_sdist_only_for_sdist_commands(
        self,
        tmp_path: Path,
    ):
        from benchmarks.utils.build_utils import _patch_sdk_sdist_build

        cached_sdist = tmp_path / "openhands-sdk.tar.gz"
        cached_sdist.write_text("cached", encoding="utf-8")
        forwarded_calls: list[tuple[list[str], str | None]] = []

        def original_run(cmd: list[str], cwd: str | None = None):
            forwarded_calls.append((cmd, cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="forwarded", stderr="")

        sdk_build_module = SimpleNamespace(_run=original_run)
        sdist_cmd = ["uv", "build", "--sdist", "--out-dir", str(tmp_path / "out")]
        other_cmd = ["docker", "buildx", "build"]

        with _patch_sdk_sdist_build(sdk_build_module, cached_sdist):
            result = sdk_build_module._run(sdist_cmd, cwd="/sdk")
            assert result.returncode == 0
            assert (tmp_path / "out" / cached_sdist.name).read_text(
                encoding="utf-8"
            ) == "cached"

            forwarded = sdk_build_module._run(other_cmd, cwd="/sdk")
            assert forwarded.stdout == "forwarded"

        assert sdk_build_module._run is original_run
        assert forwarded_calls == [(other_cmd, "/sdk")]
