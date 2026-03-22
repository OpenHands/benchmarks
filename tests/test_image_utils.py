"""Tests for image_utils and build_utils helper functions.

Tests cover local_image_exists(), create_docker_workspace(), and ensure_local_image()
which centralize Docker image detection and build logic across all benchmarks.
"""

import contextlib
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
                "build_with_telemetry",
                return_value=MagicMock(
                    tags=["ghcr.io/openhands/eval-agent-server:abcdef0-mytag"],
                    telemetry=MagicMock(
                        build_context_seconds=1.5,
                        buildx_wall_clock_seconds=12.0,
                        cleanup_seconds=0.2,
                        cache_import_seconds=3.0,
                        cache_import_miss_count=1,
                        cache_export_seconds=4.0,
                        image_export_seconds=5.0,
                        push_layers_seconds=2.5,
                        export_manifest_seconds=0.7,
                        cached_step_count=6,
                    ),
                ),
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
        assert result.sdk_cache_export_seconds == 4.0
        assert result.sdk_image_export_seconds == 5.0
        assert result.sdk_cache_import_miss_count == 1
        mock_exists.assert_not_called()
        mock_build.assert_called_once()

    @patch("benchmarks.utils.build_utils.remote_image_exists", return_value=False)
    def test_build_image_failure_preserves_sdk_telemetry(self, mock_exists):
        from benchmarks.utils.build_utils import build_image
        from openhands.agent_server.docker import build as sdk_build_module

        telemetry = MagicMock(
            build_context_seconds=2.0,
            buildx_wall_clock_seconds=30.0,
            cleanup_seconds=0.3,
            cache_import_seconds=7.0,
            cache_import_miss_count=2,
            cache_export_seconds=0.0,
            image_export_seconds=0.0,
            push_layers_seconds=0.0,
            export_manifest_seconds=0.0,
            cached_step_count=4,
        )
        failure = sdk_build_module.BuildCommandError(
            1,
            ["docker", "buildx", "build"],
            output="stdout failure",
            stderr="stderr failure",
            telemetry=telemetry,
        )

        with (
            patch(
                "benchmarks.utils.build_utils._get_sdk_submodule_info",
                return_value=("main", "abcdef0", "1.0.0"),
            ),
            patch.object(
                sdk_build_module,
                "build_with_telemetry",
                side_effect=failure,
            ),
        ):
            result = build_image(
                base_image="base:latest",
                target_image="ghcr.io/openhands/eval-agent-server",
                custom_tag="mytag",
                push=True,
            )

        assert result.status == "failed"
        assert result.sdk_build_context_seconds == 2.0
        assert result.sdk_buildx_wall_clock_seconds == 30.0
        assert result.sdk_cache_import_seconds == 7.0
        assert result.sdk_cache_import_miss_count == 2
        assert mock_exists.call_count == 3

    def test_build_parser_accepts_force_build(self):
        from benchmarks.utils.build_utils import get_build_parser

        args = get_build_parser().parse_args(["--force-build"])

        assert args.force_build is True


class TestBuildBatchSizeConfig:
    def test_build_parser_accepts_build_batch_size(self):
        from benchmarks.utils.build_utils import get_build_parser

        args = get_build_parser().parse_args(["--build-batch-size", "50"])

        assert args.build_batch_size == 50

    @patch.dict(os.environ, {"BUILD_BATCH_SIZE": "99"})
    def test_build_all_images_prefers_explicit_batch_size_over_env(
        self,
        tmp_path: Path,
    ):
        from benchmarks.utils import build_utils

        seen_batches: list[list[str]] = []

        class FakeFuture:
            def __init__(self, result: BuildOutput):
                self._result = result

            def result(self) -> BuildOutput:
                return self._result

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                self._batch: list[str] = []

            def __enter__(self):
                seen_batches.append(self._batch)
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, **kwargs):
                self._batch.append(kwargs["base_image"])
                return FakeFuture(
                    BuildOutput(
                        base_image=kwargs["base_image"],
                        tags=[f"tag:{kwargs['base_image']}"],
                        error=None,
                    )
                )

        with (
            patch.object(build_utils, "ProcessPoolExecutor", FakeExecutor),
            patch.object(
                build_utils, "as_completed", side_effect=lambda futures: futures
            ),
            patch.object(build_utils, "buildkit_disk_usage", return_value=(0, 0)),
            patch.object(build_utils, "maybe_prune_buildkit_cache", return_value=False),
        ):
            exit_code = build_utils.build_all_images(
                base_images=["base-1", "base-2", "base-3"],
                target="source-minimal",
                build_dir=tmp_path,
                build_batch_size=2,
            )

        assert exit_code == 0
        assert seen_batches == [["base-1", "base-2"], ["base-3"]]


class TestCachedSdistReuse:
    def test_build_image_passes_cached_sdist_to_sdk_build_module(
        self,
        tmp_path: Path,
    ):
        from benchmarks.utils.build_utils import build_image
        from openhands.agent_server.docker import build as sdk_build_module

        cached_sdist = tmp_path / "openhands-sdk.tar.gz"
        cached_sdist.write_text("cached", encoding="utf-8")
        captured = {}

        def fake_build(opts):
            captured["prebuilt_sdist"] = opts.prebuilt_sdist
            return MagicMock(
                tags=["integration:test"],
                telemetry=MagicMock(),
            )

        with (
            patch(
                "benchmarks.utils.build_utils.remote_image_exists", return_value=False
            ),
            patch(
                "benchmarks.utils.build_utils._get_sdk_submodule_info",
                return_value=("main", "abcdef0", "1.0.0"),
            ),
            patch.object(
                sdk_build_module, "build_with_telemetry", side_effect=fake_build
            ),
        ):
            result = build_image(
                base_image="base:latest",
                target_image="ghcr.io/openhands/eval-agent-server",
                custom_tag="mytag",
                cached_sdist=cached_sdist,
            )

        assert result.error is None
        assert result.tags == ["integration:test"]
        assert captured["prebuilt_sdist"] == cached_sdist

    def test_build_all_images_passes_cached_sdist_to_workers(self, tmp_path: Path):
        from benchmarks.utils import build_utils

        cached_sdist = tmp_path / "openhands-sdk.tar.gz"
        cached_sdist.write_text("cached", encoding="utf-8")
        submitted_kwargs: list[dict] = []

        @contextlib.contextmanager
        def fake_prepare_cached_sdist():
            yield cached_sdist

        class FakeFuture:
            def __init__(self, result: BuildOutput):
                self._result = result

            def result(self) -> BuildOutput:
                return self._result

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, **kwargs):
                submitted_kwargs.append(kwargs)
                return FakeFuture(
                    BuildOutput(
                        base_image=kwargs["base_image"],
                        tags=[f"tag:{kwargs['base_image']}"],
                        error=None,
                    )
                )

        with (
            patch.object(
                build_utils,
                "_prepare_cached_sdist",
                side_effect=fake_prepare_cached_sdist,
            ),
            patch.object(build_utils, "ProcessPoolExecutor", FakeExecutor),
            patch.object(
                build_utils, "as_completed", side_effect=lambda futures: futures
            ),
            patch.object(build_utils, "buildkit_disk_usage", return_value=(0, 0)),
            patch.object(build_utils, "maybe_prune_buildkit_cache", return_value=False),
        ):
            exit_code = build_utils.build_all_images(
                base_images=["base-1", "base-2"],
                target="source-minimal",
                build_dir=tmp_path,
            )

        assert exit_code == 0
        assert [kwargs["cached_sdist"] for kwargs in submitted_kwargs] == [
            cached_sdist,
            cached_sdist,
        ]


class TestBuildAllImagesThroughputLogging:
    @patch("benchmarks.utils.build_utils.logger.info")
    @patch(
        "benchmarks.utils.build_utils.time.monotonic",
        side_effect=[100.0, 110.0, 130.0, 145.0],
    )
    def test_throughput_logs_count_only_built_images(
        self,
        _mock_monotonic,
        mock_logger_info,
        tmp_path: Path,
    ):
        from benchmarks.utils import build_utils

        @contextlib.contextmanager
        def fake_prepare_cached_sdist():
            yield None

        class FakeFuture:
            def __init__(self, result: BuildOutput):
                self._result = result

            def result(self) -> BuildOutput:
                return self._result

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, **kwargs):
                base = kwargs["base_image"]
                if base == "base-2":
                    result = BuildOutput(
                        base_image=base,
                        tags=[f"tag:{base}"],
                        error=None,
                        status="skipped_remote_exists",
                        skip_reason="remote_image_exists",
                        duration_seconds=1.0,
                    )
                else:
                    result = BuildOutput(
                        base_image=base,
                        tags=[f"tag:{base}"],
                        error=None,
                        status="built",
                        duration_seconds=10.0,
                    )
                return FakeFuture(result)

        with (
            patch.object(
                build_utils,
                "_prepare_cached_sdist",
                side_effect=fake_prepare_cached_sdist,
            ),
            patch.object(build_utils, "ProcessPoolExecutor", FakeExecutor),
            patch.object(
                build_utils, "as_completed", side_effect=lambda futures: futures
            ),
            patch.object(build_utils, "buildkit_disk_usage", return_value=(0, 0)),
            patch.object(build_utils, "maybe_prune_buildkit_cache", return_value=False),
        ):
            exit_code = build_utils.build_all_images(
                base_images=["base-1", "base-2", "base-3"],
                target="source-minimal",
                build_dir=tmp_path,
            )

        assert exit_code == 0

        info_logs = [
            call.args[0] % call.args[1:]
            for call in mock_logger_info.call_args_list
            if call.args and isinstance(call.args[0], str)
        ]

        assert (
            "Finished batch 1/1 in 20.0s: built=2 skipped=1 failed=0 throughput=360.0 built images/hour"
            in info_logs
        )
        assert (
            "Done in 45.0s. Built=2 Skipped=1 Failed=0 Retried=0 Throughput=160.0 built images/hour "
            f"Manifest={tmp_path / 'manifest.jsonl'} Summary={tmp_path / 'build-summary.json'}"
            in info_logs
        )


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
