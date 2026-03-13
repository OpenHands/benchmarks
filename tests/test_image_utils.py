"""Tests for image_utils and build_utils helper functions.

Tests cover local_image_exists(), create_docker_workspace(), and ensure_local_image()
which centralize Docker image detection and build logic across all benchmarks.
"""

import os
import subprocess
from pathlib import Path
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


class TestBuildImageTelemetry:
    @patch("benchmarks.utils.build_utils.remote_image_exists", return_value=True)
    def test_remote_skip_sets_status_and_skip_reason(self, mock_exists):
        from benchmarks.utils.build_utils import build_image

        with patch(
            "benchmarks.utils.build_utils._get_sdk_submodule_info",
            return_value=("main", "abcdef0", "1.0.0"),
        ):
            result = build_image(
                base_image="base:latest",
                target_image="ghcr.io/openhands/eval-agent-server",
                custom_tag="mytag",
                push=True,
            )

        assert result.status == "skipped_remote_exists"
        assert result.skip_reason == "remote_image_exists"
        assert result.tags == [
            "ghcr.io/openhands/eval-agent-server:abcdef0-mytag-source-minimal"
        ]
        assert result.error is None
        assert result.remote_check_seconds is not None
        assert result.build_seconds == 0.0
        mock_exists.assert_called_once()


class TestBuildWithLoggingTelemetry:
    @patch("benchmarks.utils.build_utils.maybe_reset_buildkit")
    @patch("benchmarks.utils.build_utils.time.monotonic", side_effect=[100.0, 109.5])
    @patch("benchmarks.utils.build_utils.build_image")
    def test_successful_build_records_timing_fields(
        self, mock_build, _mock_monotonic, _mock_reset, tmp_path: Path
    ):
        from benchmarks.utils.build_utils import _build_with_logging

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=["server:v1"],
            status="built",
            remote_check_seconds=1.25,
            build_seconds=7.5,
        )

        result = _build_with_logging(
            log_dir=tmp_path,
            base_image="base:latest",
            target_image="server",
        )

        assert result.status == "built"
        assert result.attempt_count == 1
        assert result.remote_check_seconds == 1.25
        assert result.build_seconds == 7.5
        assert result.duration_seconds == 9.5
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.log_path is not None

    @patch("benchmarks.utils.build_utils.maybe_reset_buildkit")
    @patch("benchmarks.utils.build_utils.time.monotonic", side_effect=[10.0, 14.0])
    @patch("benchmarks.utils.build_utils.build_image")
    def test_failed_build_still_records_timing_fields(
        self, mock_build, _mock_monotonic, mock_reset, tmp_path: Path
    ):
        from benchmarks.utils.build_utils import _build_with_logging

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=[],
            error="boom",
            status="failed",
            remote_check_seconds=0.5,
            build_seconds=2.0,
        )

        result = _build_with_logging(
            log_dir=tmp_path,
            base_image="base:latest",
            target_image="server",
            max_retries=1,
        )

        assert result.status == "failed"
        assert result.attempt_count == 1
        assert result.remote_check_seconds == 0.5
        assert result.build_seconds == 2.0
        assert result.duration_seconds == 4.0
        mock_reset.assert_called_once()

    @patch("benchmarks.utils.build_utils.time.sleep")
    @patch("benchmarks.utils.build_utils.maybe_reset_buildkit")
    @patch("benchmarks.utils.build_utils.time.monotonic", side_effect=[50.0, 61.0])
    @patch("benchmarks.utils.build_utils.build_image")
    def test_retry_attempts_accumulate_timing_across_attempts(
        self,
        mock_build,
        _mock_monotonic,
        mock_reset,
        _mock_sleep,
        tmp_path: Path,
    ):
        from benchmarks.utils.build_utils import _build_with_logging

        mock_build.side_effect = [
            BuildOutput(
                base_image="base:latest",
                tags=[],
                error="first failure",
                status="failed",
                remote_check_seconds=1.0,
                build_seconds=2.5,
            ),
            BuildOutput(
                base_image="base:latest",
                tags=["server:v1"],
                status="built",
                remote_check_seconds=0.75,
                build_seconds=3.25,
            ),
        ]

        result = _build_with_logging(
            log_dir=tmp_path,
            base_image="base:latest",
            target_image="server",
            max_retries=2,
        )

        assert result.status == "built"
        assert result.attempt_count == 2
        assert result.remote_check_seconds == 1.75
        assert result.build_seconds == 5.75
        assert result.duration_seconds == 11.0
        mock_reset.assert_called_once()

    @patch("benchmarks.utils.build_utils.maybe_reset_buildkit")
    @patch(
        "benchmarks.utils.build_utils.time.monotonic",
        side_effect=[200.0, 204.0, 206.5, 210.0],
    )
    @patch("benchmarks.utils.build_utils.build_image")
    def test_post_build_hook_timing_is_tracked(
        self, mock_build, _mock_monotonic, _mock_reset, tmp_path: Path
    ):
        from benchmarks.utils.build_utils import _build_with_logging

        mock_build.return_value = BuildOutput(
            base_image="base:latest",
            tags=["server:v1"],
            status="built",
            remote_check_seconds=0.25,
            build_seconds=4.0,
        )

        def post_build_fn(result: BuildOutput, push: bool) -> BuildOutput:
            assert push is False
            return result

        result = _build_with_logging(
            log_dir=tmp_path,
            base_image="base:latest",
            target_image="server",
            post_build_fn=post_build_fn,
        )

        assert result.status == "built"
        assert result.post_build_seconds == 2.5
        assert result.build_seconds == 4.0
        assert result.remote_check_seconds == 0.25
        assert result.duration_seconds == 10.0
