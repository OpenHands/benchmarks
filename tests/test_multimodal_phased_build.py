"""Tests for the SWE-bench Multimodal three-phase build pipeline.

Mirrors test_phased_build.py but exercises the multimodal entry point,
which reuses the shared phased-build functions from build_base_images.
"""

from unittest.mock import patch

from benchmarks.utils.build_utils import BuildOutput


# ---------------------------------------------------------------------------
# extract_custom_tag
# ---------------------------------------------------------------------------


class TestExtractCustomTag:
    def test_regular_swebench_image(self):
        from benchmarks.swebenchmultimodal.build_images import extract_custom_tag

        tag = extract_custom_tag(
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11333:latest"
        )
        assert tag == "sweb.eval.x86_64.django_1776_django-11333"

    def test_multimodal_image_format(self):
        from benchmarks.swebenchmultimodal.build_images import extract_custom_tag

        tag = extract_custom_tag(
            "docker.io/swebench/sweb.mm.eval.x86_64.openlayers_1776_openlayers-12172:latest"
        )
        assert tag == "sweb.mm.eval.x86_64.openlayers_1776_openlayers-12172"

    def test_no_tag_suffix(self):
        from benchmarks.swebenchmultimodal.build_images import extract_custom_tag

        tag = extract_custom_tag("docker.io/swebench/sweb.eval.x86_64.foo_1776_bar")
        assert tag == "sweb.eval.x86_64.foo_1776_bar"


# ---------------------------------------------------------------------------
# get_official_docker_image
# ---------------------------------------------------------------------------


class TestGetOfficialDockerImage:
    def test_produces_regular_swebench_image(self):
        from benchmarks.swebenchmultimodal.build_images import get_official_docker_image

        image = get_official_docker_image("django__django-11333")
        assert (
            image
            == "docker.io/swebench/sweb.eval.x86_64.django_1776_django-11333:latest"
        )

    def test_lowercased(self):
        from benchmarks.swebenchmultimodal.build_images import get_official_docker_image

        image = get_official_docker_image("MyRepo__MyName-123")
        assert image == image.lower()


# ---------------------------------------------------------------------------
# collect_unique_base_images
# ---------------------------------------------------------------------------


class TestCollectUniqueBaseImages:
    @patch("benchmarks.swebenchmultimodal.build_images.get_dataset")
    def test_deduplicates_and_sorts(self, mock_get_dataset):
        import pandas as pd

        from benchmarks.swebenchmultimodal.build_images import (
            collect_unique_base_images,
        )

        mock_get_dataset.return_value = pd.DataFrame(
            {
                "instance_id": [
                    "django__django-11333",
                    "django__django-11333",  # duplicate
                    "astropy__astropy-12345",
                ],
            }
        )

        images = collect_unique_base_images("ds", "test", 0)
        assert len(images) == 2  # deduplicated
        assert images == sorted(images)  # sorted

    @patch("benchmarks.swebenchmultimodal.build_images.get_dataset")
    def test_passes_selected_instances_file(self, mock_get_dataset):
        import pandas as pd

        from benchmarks.swebenchmultimodal.build_images import (
            collect_unique_base_images,
        )

        mock_get_dataset.return_value = pd.DataFrame(
            {"instance_id": ["django__django-11333"]}
        )

        collect_unique_base_images(
            "ds", "test", 0, selected_instances_file="/tmp/ids.txt"
        )
        mock_get_dataset.assert_called_once_with(
            dataset_name="ds",
            split="test",
            eval_limit=None,
            selected_instances_file="/tmp/ids.txt",
        )


# ---------------------------------------------------------------------------
# Phase orchestration (multimodal build_images.main)
# ---------------------------------------------------------------------------


class TestMultimodalPhasedOrchestration:
    """Test that the multimodal main() orchestrates the three phases correctly.

    These tests mock the shared phased-build functions (build_builder_image,
    build_all_base_images, assemble_all_agent_images) which are tested
    independently in test_phased_build.py.
    """

    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=0,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebenchmultimodal.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_happy_path_all_phases(
        self, _collect, mock_builder, mock_bases, mock_assemble
    ):
        from benchmarks.swebenchmultimodal.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        rc = main(["--dataset", "test-ds", "--split", "dev"])

        assert rc == 0
        mock_builder.assert_called_once()
        mock_bases.assert_called_once()
        mock_assemble.assert_called_once()
        # Verify builder tag is passed to assembly phase
        assert mock_assemble.call_args.kwargs["builder_tag"] == "builder:abc"

    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebenchmultimodal.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_builder_failure_aborts_early(self, _collect, mock_builder):
        from benchmarks.swebenchmultimodal.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=[],
            error="build failed",
        )

        rc = main(["--dataset", "test-ds", "--split", "dev"])
        assert rc == 1

    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=1
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebenchmultimodal.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_base_failure_aborts_before_assembly(self, _collect, mock_builder, _bases):
        from benchmarks.swebenchmultimodal.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        rc = main(["--dataset", "test-ds", "--split", "dev"])
        assert rc == 1

    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=1,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebenchmultimodal.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_assembly_failure_returns_nonzero(
        self, _collect, mock_builder, _bases, _assemble
    ):
        from benchmarks.swebenchmultimodal.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        rc = main(["--dataset", "test-ds", "--split", "dev"])
        assert rc == 1

    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=0,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebenchmultimodal.build_images.collect_unique_base_images",
        return_value=["img-a", "img-b"],
    )
    def test_select_arg_forwarded(self, mock_collect, mock_builder, _bases, _assemble):
        from benchmarks.swebenchmultimodal.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        main(["--dataset", "ds", "--split", "dev", "--select", "/tmp/ids.txt"])
        mock_collect.assert_called_once_with("ds", "dev", 0, "/tmp/ids.txt")

    @patch(
        "benchmarks.swebench.build_base_images.assemble_all_agent_images",
        return_value=0,
    )
    @patch(
        "benchmarks.swebench.build_base_images.build_all_base_images", return_value=0
    )
    @patch("benchmarks.swebench.build_base_images.build_builder_image")
    @patch(
        "benchmarks.swebenchmultimodal.build_images.collect_unique_base_images",
        return_value=["img-a"],
    )
    def test_force_build_forwarded(self, _collect, mock_builder, _bases, mock_assemble):
        from benchmarks.swebenchmultimodal.build_images import main

        mock_builder.return_value = BuildOutput(
            base_image="builder",
            tags=["builder:abc"],
            error=None,
        )

        main(["--dataset", "ds", "--split", "dev", "--force-build"])
        assert mock_builder.call_args.kwargs["force_build"] is True
        assert _bases.call_args.kwargs["force_build"] is True
        assert mock_assemble.call_args.kwargs["force_build"] is True


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


class TestMultimodalParser:
    def test_defaults(self):
        from benchmarks.swebenchmultimodal.build_images import get_parser

        parser = get_parser()
        args = parser.parse_args([])
        assert "Multimodal" in args.dataset
        assert args.split == "dev"
        assert args.target == "source-minimal"
        assert args.push is False
        assert args.force_build is False
        assert args.n_limit == 0
        assert args.select is None
