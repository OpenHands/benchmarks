"""Tests for benchmarks.swebench.build_images local grading image helpers.

Covers prepare_local_grading_image() and
prepare_local_grading_images_for_predictions(), which let `swebench-eval
--no-modal` reuse already-built local eval-agent-server images as SWE-Bench
grading images (with ENTRYPOINT cleared) instead of pulling/building the
official images from Docker Hub.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.utils.build_utils import BuildOutput


class TestPrepareLocalGradingImage:
    @patch("benchmarks.swebench.build_images.local_image_exists", return_value=False)
    def test_errors_when_agent_image_not_local(self, _mock_exists):
        from benchmarks.swebench.build_images import prepare_local_grading_image

        result = prepare_local_grading_image(
            agent_image="ghcr.io/openhands/eval-agent-server:abc-sweb.eval.x86_64.foo_1776_bar-source-minimal",
            grading_image="swebench/sweb.eval.x86_64.foo_1776_bar:latest",
        )
        assert result.error is not None
        assert "not found locally" in result.error
        assert result.tags == []

    @patch("benchmarks.swebench.build_images.subprocess.run")
    @patch("benchmarks.swebench.build_images.local_image_exists", return_value=True)
    def test_builds_with_expected_args(self, _mock_exists, mock_run):
        """Uses plain `docker build` (not `docker buildx build`): buildx
        routes through whichever builder is currently active, and a
        docker-container-driver builder can't see images that only exist in
        the host daemon's local store. See prepare_local_grading_image()'s
        docstring.
        """
        from benchmarks.swebench.build_images import prepare_local_grading_image

        agent_image = (
            "ghcr.io/openhands/eval-agent-server:"
            "abc-sweb.eval.x86_64.foo_1776_bar-source-minimal"
        )
        grading_image = "swebench/sweb.eval.x86_64.foo_1776_bar:latest"
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        result = prepare_local_grading_image(agent_image, grading_image)

        assert result.error is None
        assert result.tags == [grading_image]
        (cmd,), kwargs = mock_run.call_args
        assert cmd[:2] == ["docker", "build"]
        assert "buildx" not in cmd
        assert f"SDK_IMAGE={agent_image}" in cmd
        assert grading_image in cmd
        assert kwargs.get("capture_output") is True

    @patch("benchmarks.swebench.build_images.subprocess.run")
    @patch("benchmarks.swebench.build_images.local_image_exists", return_value=True)
    def test_records_docker_build_failure(self, _mock_exists, mock_run):
        from benchmarks.swebench.build_images import prepare_local_grading_image

        agent_image = (
            "ghcr.io/openhands/eval-agent-server:"
            "abc-sweb.eval.x86_64.foo_1776_bar-source-minimal"
        )
        grading_image = "swebench/sweb.eval.x86_64.foo_1776_bar:latest"
        mock_run.return_value = SimpleNamespace(
            returncode=1, stdout="", stderr="something went wrong"
        )

        result = prepare_local_grading_image(agent_image, grading_image)

        assert result.error is not None
        assert "something went wrong" in result.error
        assert result.tags == []


class TestPrepareLocalGradingImagesForPredictions:
    def _write_predictions(self, instance_ids: list[str]) -> Path:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for iid in instance_ids:
            f.write(json.dumps({"instance_id": iid, "model_patch": "diff"}) + "\n")
        f.close()
        return Path(f.name)

    @patch("benchmarks.swebench.build_images.get_phased_image_tag_prefix")
    @patch("benchmarks.swebench.build_images.prepare_local_grading_image")
    @patch("benchmarks.swebench.build_images.local_image_exists")
    def test_skips_instance_with_official_image_already_local(
        self, mock_exists, mock_prepare, mock_prefix
    ):
        from benchmarks.swebench.build_images import (
            prepare_local_grading_images_for_predictions,
        )

        mock_prefix.return_value = "abc123-def456"
        # Official grading image already exists locally -> nothing to do.
        mock_exists.return_value = True

        predictions = self._write_predictions(["django__django-11333"])
        try:
            results = prepare_local_grading_images_for_predictions(predictions)
        finally:
            predictions.unlink()

        assert results == {"django__django-11333": False}
        mock_prepare.assert_not_called()

    @patch("benchmarks.swebench.build_images.get_phased_image_tag_prefix")
    @patch("benchmarks.swebench.build_images.prepare_local_grading_image")
    @patch("benchmarks.swebench.build_images.local_image_exists")
    def test_skips_instance_with_no_local_agent_image(
        self, mock_exists, mock_prepare, mock_prefix
    ):
        from benchmarks.swebench.build_images import (
            prepare_local_grading_images_for_predictions,
        )

        mock_prefix.return_value = "abc123-def456"
        # Neither the official grading image nor the agent-server image
        # exist locally -> nothing to prepare, fall back to SWE-Bench.
        mock_exists.return_value = False

        predictions = self._write_predictions(["django__django-11333"])
        try:
            results = prepare_local_grading_images_for_predictions(predictions)
        finally:
            predictions.unlink()

        assert results == {"django__django-11333": False}
        mock_prepare.assert_not_called()

    @patch("benchmarks.swebench.build_images.get_phased_image_tag_prefix")
    @patch("benchmarks.swebench.build_images.prepare_local_grading_image")
    @patch("benchmarks.swebench.build_images.local_image_exists")
    def test_prepares_instance_with_local_agent_image(
        self, mock_exists, mock_prepare, mock_prefix
    ):
        from benchmarks.swebench.build_images import (
            prepare_local_grading_images_for_predictions,
        )

        mock_prefix.return_value = "abc123-def456"
        grading_image = (
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11333:latest"
        )
        agent_image = (
            "ghcr.io/openhands/eval-agent-server:"
            "abc123-def456-sweb.eval.x86_64.django_1776_django-11333-source-minimal"
        )
        # Official image missing locally, but the agent-server image is
        # present -> should prepare.
        mock_exists.side_effect = lambda image: image == agent_image
        mock_prepare.return_value = BuildOutput(
            base_image=agent_image,
            tags=[grading_image],
            error=None,
        )

        predictions = self._write_predictions(["django__django-11333"])
        try:
            results = prepare_local_grading_images_for_predictions(predictions)
        finally:
            predictions.unlink()

        assert results == {"django__django-11333": True}
        mock_prepare.assert_called_once_with(agent_image, grading_image)

    @patch("benchmarks.swebench.build_images.get_phased_image_tag_prefix")
    @patch("benchmarks.swebench.build_images.prepare_local_grading_image")
    @patch("benchmarks.swebench.build_images.local_image_exists")
    def test_records_failure_without_raising(
        self, mock_exists, mock_prepare, mock_prefix
    ):
        from benchmarks.swebench.build_images import (
            prepare_local_grading_images_for_predictions,
        )

        mock_prefix.return_value = "abc123-def456"
        agent_image = (
            "ghcr.io/openhands/eval-agent-server:"
            "abc123-def456-sweb.eval.x86_64.django_1776_django-11333-source-minimal"
        )
        mock_exists.side_effect = lambda image: image == agent_image
        mock_prepare.return_value = BuildOutput(
            base_image=agent_image,
            tags=[],
            error="docker build failed",
        )

        predictions = self._write_predictions(["django__django-11333"])
        try:
            results = prepare_local_grading_images_for_predictions(predictions)
        finally:
            predictions.unlink()

        # Failure to prepare shouldn't raise; caller falls back to SWE-Bench.
        assert results == {"django__django-11333": False}
