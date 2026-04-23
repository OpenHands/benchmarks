"""Tests for the phased benchmark image build (build_base_images + build_images)."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

from benchmarks.utils.build_utils import BuildOutput


# 7-char lowercase hex hash prefix expected in base image tags.
_HASH_RE = re.compile(r":([0-9a-f]{7})-")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_proc(stdout="", stderr=""):
    """Fake subprocess.CompletedProcess with returncode 0."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=stderr
    )


def _fail_proc(stderr="build error", code=1):
    """Fake subprocess.CompletedProcess with non-zero returncode."""
    return subprocess.CompletedProcess(
        args=[], returncode=code, stdout="", stderr=stderr
    )


def _timeout_exc(stdout="", stderr=""):
    """Fake subprocess.TimeoutExpired with optional partial output."""
    return subprocess.TimeoutExpired(
        cmd=["docker"],
        timeout=1,
        output=stdout,
        stderr=stderr,
    )


# Production code uses ProcessPoolExecutor for true parallelism across builds.
# Tests substitute ThreadPoolExecutor to avoid pickling issues with mocks.
def _thread_pool(**kw):
    return ThreadPoolExecutor(**kw)


class TestSWEBenchBuildImages:
    def test_parser_does_not_accept_agent_type(self):
        from benchmarks.swebench.build_images import get_parser

        parser = get_parser()

        with patch("argparse.ArgumentParser.exit", side_effect=SystemExit) as mock_exit:
            try:
                parser.parse_args(["--agent-type", "acp-claude"])
            except SystemExit:
                pass

        mock_exit.assert_called_once()

    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=0,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_builder_image",
        return_value=BuildOutput(
            base_image="builder", tags=["builder:tag"], error=None
        ),
    )
    @patch(
        "benchmarks.swebench.build_images.collect_unique_base_images",
        return_value=[
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest"
        ],
    )
    @patch(
        "benchmarks.swebench.build_images.default_build_output_dir",
        return_value="build-dir",
    )
    def test_main_builds_without_agent_type(
        self,
        _build_dir,
        collect_unique_base_images,
        build_builder_image,
        build_all_base_images,
        assemble_all_agent_images,
    ):
        from benchmarks.swebench.build_images import main

        rc = main(["--dataset", "dataset", "--split", "test"])

        assert rc == 0
        collect_unique_base_images.assert_called_once_with(
            "dataset",
            "test",
            0,
            None,
        )
        build_builder_image.assert_called_once_with(push=False, force_build=False)
        build_all_base_images.assert_called_once()
        assemble_all_agent_images.assert_called_once()


# ---------------------------------------------------------------------------
# build_base_image: basic success / failure / skip
# ---------------------------------------------------------------------------


class TestBuildBaseImage:
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
        return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
    )
    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists", return_value=True
    )
    def test_skips_when_remote_exists(self, _exists, _dockerfile):
        from benchmarks.swebench.build_base_images import build_base_image

        result = build_base_image(
            "ubuntu:22.04", "custom-tag", push=False, content_hash="abc1234"
        )
        assert result.error is None
        assert len(result.tags) == 1
        assert "custom-tag" in result.tags[0]

    @patch(
        "benchmarks.swebench.build_base_images.subprocess.run", return_value=_ok_proc()
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
        return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
    )
    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists",
        return_value=False,
    )
    def test_success(self, _exists, _dockerfile, mock_run):
        from benchmarks.swebench.build_base_images import build_base_image

        result = build_base_image(
            "ubuntu:22.04", "custom-tag", push=True, content_hash="abc1234"
        )
        assert result.error is None
        assert len(result.tags) == 1
        cmd = mock_run.call_args[0][0]
        assert "--push" in cmd

    @patch(
        "benchmarks.swebench.build_base_images.subprocess.run",
        return_value=_fail_proc(),
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
        return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
    )
    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists",
        return_value=False,
    )
    def test_failure_returns_error(self, _exists, _dockerfile, _run):
        from benchmarks.swebench.build_base_images import build_base_image

        result = build_base_image("ubuntu:22.04", "custom-tag", content_hash="abc1234")
        assert result.error is not None
        assert result.tags == []

    @patch(
        "benchmarks.swebench.build_base_images.subprocess.run",
        side_effect=_timeout_exc(stderr="stalled build"),
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
        return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
    )
    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists",
        return_value=False,
    )
    def test_timeout_returns_error(self, _exists, _dockerfile, _run):
        from benchmarks.swebench.build_base_images import build_base_image

        result = build_base_image("ubuntu:22.04", "custom-tag", content_hash="abc1234")
        assert result.tags == []
        assert result.error is not None
        assert "timed out" in result.error

    @patch(
        "benchmarks.swebench.build_base_images.subprocess.run", return_value=_ok_proc()
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
        return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
    )
    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists",
        return_value=True,
    )
    def test_force_build_bypasses_remote_exists(self, _exists, _dockerfile, mock_run):
        from benchmarks.swebench.build_base_images import build_base_image

        result = build_base_image(
            "ubuntu:22.04",
            "custom-tag",
            force_build=True,
            content_hash="abc1234",
        )
        assert result.error is None
        assert len(result.tags) == 1
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# _build_base_with_logging: retry logic
# ---------------------------------------------------------------------------


class TestBuildBaseWithLoggingRetry:
    @patch("benchmarks.swebench.build_base_images.build_base_image")
    @patch("benchmarks.swebench.build_base_images.capture_output")
    def test_retries_on_failure_then_succeeds(self, mock_capture, mock_build, tmp_path):
        from benchmarks.swebench.build_base_images import _build_base_with_logging

        log_path = tmp_path / "log.txt"
        log_path.touch()
        mock_capture.return_value.__enter__ = MagicMock(return_value=log_path)
        mock_capture.return_value.__exit__ = MagicMock(return_value=False)

        fail_result = BuildOutput(base_image="img", tags=[], error="transient error")
        ok_result = BuildOutput(base_image="img", tags=["tag:1"], error=None)
        mock_build.side_effect = [fail_result, ok_result]

        result = _build_base_with_logging(
            log_dir=tmp_path,
            base_image="img",
            custom_tag="tag",
            max_retries=3,
            content_hash="abc1234",
        )
        assert result.error is None
        assert result.tags == ["tag:1"]
        assert mock_build.call_count == 2

    @patch("benchmarks.swebench.build_base_images.build_base_image")
    @patch("benchmarks.swebench.build_base_images.capture_output")
    def test_exhausts_retries(self, mock_capture, mock_build, tmp_path):
        from benchmarks.swebench.build_base_images import _build_base_with_logging

        log_path = tmp_path / "log.txt"
        log_path.touch()
        mock_capture.return_value.__enter__ = MagicMock(return_value=log_path)
        mock_capture.return_value.__exit__ = MagicMock(return_value=False)

        fail_result = BuildOutput(base_image="img", tags=[], error="permanent error")
        mock_build.return_value = fail_result

        result = _build_base_with_logging(
            log_dir=tmp_path,
            base_image="img",
            custom_tag="tag",
            max_retries=2,
            content_hash="abc1234",
        )
        assert result.error == "permanent error"
        assert mock_build.call_count == 2

    @patch("benchmarks.swebench.build_base_images.build_base_image")
    @patch("benchmarks.swebench.build_base_images.capture_output")
    def test_exception_captured_as_error(self, mock_capture, mock_build, tmp_path):
        from benchmarks.swebench.build_base_images import _build_base_with_logging

        log_path = tmp_path / "log.txt"
        log_path.touch()
        mock_capture.return_value.__enter__ = MagicMock(return_value=log_path)
        mock_capture.return_value.__exit__ = MagicMock(return_value=False)

        mock_build.side_effect = RuntimeError("docker crash")

        result = _build_base_with_logging(
            log_dir=tmp_path,
            base_image="img",
            custom_tag="tag",
            max_retries=1,
            content_hash="abc1234",
        )
        assert result.error is not None
        assert "docker crash" in result.error


# ---------------------------------------------------------------------------
# assemble_agent_image: partial push failures
# ---------------------------------------------------------------------------


class TestAssembleAgentImage:
    def test_partial_push_failure_collected(self, tmp_path):
        from benchmarks.swebench.build_base_images import assemble_agent_image

        # Create a real Dockerfile so .exists() returns True
        dockerfile = tmp_path / "Dockerfile.agent-layer"
        dockerfile.write_text("FROM scratch\n")

        with (
            patch(
                "benchmarks.swebench.build_base_images.AGENT_LAYER_DOCKERFILE",
                dockerfile,
            ),
            patch("benchmarks.swebench.build_base_images.subprocess.run") as mock_run,
        ):
            # Build succeeds, first push succeeds, second push fails
            mock_run.side_effect = [
                _ok_proc(),  # docker build
                _ok_proc(),  # docker push tag-1
                _fail_proc("push denied"),  # docker push tag-2
            ]

            result = assemble_agent_image(
                base_tag="ghcr.io/openhands/eval-base:abc",
                builder_tag="ghcr.io/openhands/eval-builder:def",
                final_tags=["tag-1", "tag-2"],
                push=True,
            )

        assert result.error is not None
        assert "Failed to push 1/2 tags" in result.error
        assert result.tags == ["tag-1"]

    def test_build_failure_returns_error(self, tmp_path):
        from benchmarks.swebench.build_base_images import assemble_agent_image

        dockerfile = tmp_path / "Dockerfile.agent-layer"
        dockerfile.write_text("FROM scratch\n")

        with (
            patch(
                "benchmarks.swebench.build_base_images.AGENT_LAYER_DOCKERFILE",
                dockerfile,
            ),
            patch("benchmarks.swebench.build_base_images.subprocess.run") as mock_run,
        ):
            mock_run.return_value = _fail_proc("docker build failed")

            result = assemble_agent_image(
                base_tag="ghcr.io/openhands/eval-base:abc",
                builder_tag="ghcr.io/openhands/eval-builder:def",
                final_tags=["tag-1"],
                push=True,
            )

        assert result.error is not None
        assert "docker build failed" in result.error
        assert result.tags == []

    def test_all_pushes_succeed(self, tmp_path):
        from benchmarks.swebench.build_base_images import assemble_agent_image

        dockerfile = tmp_path / "Dockerfile.agent-layer"
        dockerfile.write_text("FROM scratch\n")

        with (
            patch(
                "benchmarks.swebench.build_base_images.AGENT_LAYER_DOCKERFILE",
                dockerfile,
            ),
            patch("benchmarks.swebench.build_base_images.subprocess.run") as mock_run,
        ):
            mock_run.return_value = _ok_proc()

            result = assemble_agent_image(
                base_tag="ghcr.io/openhands/eval-base:abc",
                builder_tag="ghcr.io/openhands/eval-builder:def",
                final_tags=["tag-1", "tag-2"],
                push=True,
            )

        assert result.error is None
        assert result.tags == ["tag-1", "tag-2"]

    def test_missing_dockerfile_returns_error(self, tmp_path):
        from benchmarks.swebench.build_base_images import assemble_agent_image

        fake_path = tmp_path / "nonexistent" / "Dockerfile.agent-layer"
        with patch(
            "benchmarks.swebench.build_base_images.AGENT_LAYER_DOCKERFILE", fake_path
        ):
            result = assemble_agent_image(
                base_tag="base:tag",
                builder_tag="builder:tag",
                final_tags=["out:tag"],
            )
        assert result.error is not None
        assert "not found" in result.error

    def test_build_timeout_returns_error(self, tmp_path):
        from benchmarks.swebench.build_base_images import assemble_agent_image

        dockerfile = tmp_path / "Dockerfile.agent-layer"
        dockerfile.write_text("FROM scratch\n")

        with (
            patch(
                "benchmarks.swebench.build_base_images.AGENT_LAYER_DOCKERFILE",
                dockerfile,
            ),
            patch(
                "benchmarks.swebench.build_base_images.subprocess.run",
                side_effect=_timeout_exc(stderr="build stalled"),
            ),
        ):
            result = assemble_agent_image(
                base_tag="ghcr.io/openhands/eval-base:abc",
                builder_tag="ghcr.io/openhands/eval-builder:def",
                final_tags=["tag-1"],
                push=True,
            )

        assert result.tags == []
        assert result.error is not None
        assert "timed out" in result.error


# ---------------------------------------------------------------------------
# build_all_base_images: manifest writing and failure counting
# ---------------------------------------------------------------------------


class TestBuildAllBaseImages:
    @patch("benchmarks.swebench.build_base_images._build_base_with_logging")
    @patch(
        "benchmarks.swebench.build_base_images.ProcessPoolExecutor", new=_thread_pool
    )
    def test_manifest_written_and_failures_counted(self, mock_build, tmp_path):
        from benchmarks.swebench.build_base_images import build_all_base_images

        ok = BuildOutput(base_image="img-a", tags=["tag-a"], error=None)
        fail = BuildOutput(base_image="img-b", tags=[], error="boom")
        mock_build.side_effect = [ok, fail]

        rc = build_all_base_images(
            base_images=["img-a", "img-b"],
            build_dir=tmp_path,
            max_workers=1,
            max_retries=1,
        )

        assert rc == 1  # failures present
        manifest = tmp_path / "base-manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text().strip().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        errors = [r for r in records if r.get("error")]
        successes = [r for r in records if not r.get("error")]
        assert len(errors) == 1
        assert len(successes) == 1

    @patch("benchmarks.swebench.build_base_images._build_base_with_logging")
    @patch(
        "benchmarks.swebench.build_base_images.ProcessPoolExecutor", new=_thread_pool
    )
    def test_all_succeed_returns_zero(self, mock_build, tmp_path):
        from benchmarks.swebench.build_base_images import build_all_base_images

        ok = BuildOutput(base_image="img-a", tags=["tag-a"], error=None)
        mock_build.return_value = ok

        rc = build_all_base_images(
            base_images=["img-a"],
            build_dir=tmp_path,
            max_workers=1,
        )

        assert rc == 0

    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
        return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
    )
    def test_dry_run_prints_without_building(self, _dockerfile, tmp_path, capsys):
        from benchmarks.swebench.build_base_images import build_all_base_images

        rc = build_all_base_images(
            base_images=["ubuntu:22.04"],
            build_dir=tmp_path,
            dry_run=True,
        )

        assert rc == 0
        captured = capsys.readouterr()
        assert "ubuntu:22.04" in captured.out


# ---------------------------------------------------------------------------
# assemble_all_agent_images: manifest writing
# ---------------------------------------------------------------------------


class TestAssembleAllAgentImages:
    @patch("benchmarks.swebench.build_base_images._assemble_with_logging")
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_submodule_info",
        return_value=("sdk", "abc1234567", "v1"),
    )
    @patch(
        "benchmarks.swebench.build_base_images.ProcessPoolExecutor", new=_thread_pool
    )
    def test_manifest_written(self, mock_sdk, mock_assemble, tmp_path):
        from benchmarks.swebench.build_base_images import assemble_all_agent_images

        ok = BuildOutput(base_image="img-a", tags=["final-tag"], error=None)
        mock_assemble.return_value = ok

        rc = assemble_all_agent_images(
            base_images=["img-a"],
            builder_tag="builder:tag",
            build_dir=tmp_path,
            max_workers=1,
        )

        assert rc == 0
        manifest = tmp_path / "manifest.jsonl"
        assert manifest.exists()
        records = [
            json.loads(line) for line in manifest.read_text().strip().splitlines()
        ]
        assert len(records) == 1
        assert records[0]["tags"] == ["final-tag"]


# ---------------------------------------------------------------------------
# build_builder_image: uses builder_image as build_id (not "sdk-builder-stage")
# ---------------------------------------------------------------------------


class TestBuildBuilderImage:
    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists", return_value=True
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_submodule_info",
        return_value=("sdk", "abc1234567", "v1"),
    )
    def test_skipped_result_uses_builder_repo_not_stage_name(self, _sdk, _exists):
        from benchmarks.swebench.build_base_images import (
            EVAL_BUILDER_IMAGE,
            build_builder_image,
        )

        result = build_builder_image()
        # Should use the builder image repo name, not "sdk-builder-stage"
        assert result.base_image == EVAL_BUILDER_IMAGE
        assert result.error is None
        assert len(result.tags) == 1

    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists", return_value=True
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_submodule_info",
        return_value=("sdk", "abc1234567", "v1"),
    )
    @patch("benchmarks.swebench.build_base_images._get_repo_root")
    @patch("openhands.agent_server.docker.build._make_build_context")
    @patch(
        "benchmarks.swebench.build_base_images.subprocess.run", return_value=_ok_proc()
    )
    def test_force_build_bypasses_remote_exists(
        self, mock_run, mock_make_context, mock_repo_root, _sdk, _exists, tmp_path
    ):
        from benchmarks.swebench.build_base_images import build_builder_image

        ctx = tmp_path / "ctx"
        ctx.mkdir()
        (ctx / "Dockerfile").write_text("FROM scratch\n")
        mock_make_context.return_value = ctx
        mock_repo_root.return_value = tmp_path

        result = build_builder_image(force_build=True)

        assert result.error is None
        assert len(result.tags) == 1
        mock_run.assert_called_once()

    @patch(
        "benchmarks.swebench.build_base_images.remote_image_exists", return_value=False
    )
    @patch(
        "benchmarks.swebench.build_base_images._get_sdk_submodule_info",
        return_value=("sdk", "abc1234567", "v1"),
    )
    @patch("benchmarks.swebench.build_base_images._get_repo_root")
    @patch("openhands.agent_server.docker.build._make_build_context")
    @patch(
        "benchmarks.swebench.build_base_images.subprocess.run",
        side_effect=_timeout_exc(stderr="builder stalled"),
    )
    def test_timeout_returns_error(
        self, _run, mock_make_context, mock_repo_root, _sdk, _exists, tmp_path
    ):
        from benchmarks.swebench.build_base_images import build_builder_image

        ctx = tmp_path / "ctx"
        ctx.mkdir()
        (ctx / "Dockerfile").write_text("FROM scratch\n")
        mock_make_context.return_value = ctx
        mock_repo_root.return_value = tmp_path

        result = build_builder_image()

        assert result.tags == []
        assert result.error is not None
        assert "timed out" in result.error


# ---------------------------------------------------------------------------
# Phase orchestration (build_images.main)
# ---------------------------------------------------------------------------


class TestPhasedOrchestration:
    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=0,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebench.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_happy_path(self, _collect, mock_builder, mock_bases, mock_assemble):
        from benchmarks.swebench.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        rc = main(["--dataset", "test-ds", "--split", "test"])

        assert rc == 0
        mock_builder.assert_called_once()
        mock_bases.assert_called_once()
        mock_assemble.assert_called_once()
        assert mock_assemble.call_args.kwargs["builder_tag"] == "builder:abc"

    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=0,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebench.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_force_build_forwarded_to_all_phases(
        self, _collect, mock_builder, mock_bases, mock_assemble
    ):
        from benchmarks.swebench.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        rc = main(["--dataset", "test-ds", "--split", "test", "--force-build"])

        assert rc == 0
        assert mock_builder.call_args.kwargs["force_build"] is True
        assert mock_bases.call_args.kwargs["force_build"] is True
        assert mock_assemble.call_args.kwargs["force_build"] is True

    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebench.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_builder_failure_aborts(self, _collect, mock_builder):
        from benchmarks.swebench.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=[],
            error="build failed",
        )

        rc = main(["--dataset", "test-ds", "--split", "test"])
        assert rc == 1

    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=1
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebench.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_base_failure_aborts_before_assembly(self, _collect, mock_builder, _bases):
        from benchmarks.swebench.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        rc = main(["--dataset", "test-ds", "--split", "test"])
        assert rc == 1


# ---------------------------------------------------------------------------
# base_image_tag: simple tag computation
# ---------------------------------------------------------------------------


class TestBaseImageTag:
    def test_default_registry(self):
        from benchmarks.swebench.build_base_images import base_image_tag

        tag = base_image_tag("my-custom-tag", content_hash="abc1234")
        assert tag.endswith(":abc1234-my-custom-tag")
        assert _HASH_RE.search(tag), f"tag missing 7-char hex prefix: {tag}"

    def test_custom_registry(self):
        from benchmarks.swebench.build_base_images import base_image_tag

        tag = base_image_tag("abc", image="my-registry/my-repo", content_hash="abc1234")
        assert tag == "my-registry/my-repo:abc1234-abc"

    def test_hash_changes_with_dockerfile_content(self):
        from benchmarks.swebench.build_base_images import base_image_tag

        tag1 = base_image_tag("x", content_hash="aaaaaaa")
        tag2 = base_image_tag("x", content_hash="bbbbbbb")
        assert tag1 != tag2

    def test_dockerfile_content_hash_format(self):
        from benchmarks.swebench.build_base_images import dockerfile_content_hash

        with patch(
            "benchmarks.swebench.build_base_images._get_sdk_dockerfile",
            return_value=Mock(read_text=Mock(return_value="FROM ubuntu:22.04\n")),
        ):
            h = dockerfile_content_hash()

        assert re.fullmatch(r"[0-9a-f]{7}", h), f"expected 7-char hex, got {h!r}"
