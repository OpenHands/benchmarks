"""Tests for image_utils and build_utils helper functions.

Tests cover local_image_exists(), create_docker_workspace(), and ensure_local_image()
which centralize Docker image detection and build logic across all benchmarks.
"""

import os
import subprocess
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


class TestRemoteForceBuild:
    @patch("benchmarks.utils.build_utils.remote_image_exists", return_value=True)
    def test_build_image_force_build_bypasses_remote_exists(self, mock_exists):
        from benchmarks.utils.build_utils import build_image
        from openhands.agent_server.docker import build as sdk_build_module

        with (
            patch(
                "benchmarks.utils.build_utils._get_sdk_submodule_info",
                return_value=("main", "abcdef0", "1.0.0"),
            ),
            patch.object(
                sdk_build_module,
                "build",
                return_value=["ghcr.io/openhands/eval-agent-server:abcdef0-mytag"],
            ) as mock_build,
        ):
            result = build_image(
                base_image="base:latest",
                target_image="ghcr.io/openhands/eval-agent-server",
                custom_tag="mytag",
                push=True,
                force_build=True,
            )

        assert result.error is None
        assert result.tags == ["ghcr.io/openhands/eval-agent-server:abcdef0-mytag"]
        mock_exists.assert_not_called()
        mock_build.assert_called_once()

    def test_build_parser_accepts_force_build(self):
        from benchmarks.utils.build_utils import get_build_parser

        args = get_build_parser().parse_args(["--force-build"])

        assert args.force_build is True
