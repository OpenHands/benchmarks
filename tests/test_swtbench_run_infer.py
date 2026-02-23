"""Tests for SWT-bench run_infer module.

Tests verify that SWT-bench uses the same image building approach as SWE-bench,
ensuring Docker workspace mode properly builds images when SKIP_BUILD=0.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.swtbench.run_infer import (
    DEFAULT_BUILD_TARGET,
    SWTBenchEvaluation,
)
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.models import EvalInstance, EvalMetadata


class TestSWTBenchImageTagGeneration:
    """Tests for SWT-bench Docker image tag generation."""

    def test_default_build_target_is_source_minimal(self):
        """Verify SWT-bench uses source-minimal as default build target."""
        assert DEFAULT_BUILD_TARGET == "source-minimal"

    def test_swtbench_imports_from_swebench(self):
        """Verify SWT-bench imports official Docker image helpers from SWE-bench."""
        from benchmarks.swtbench.run_infer import (
            get_official_docker_image,
        )

        # Test get_official_docker_image produces correct format
        instance_id = "django__django-12345"
        image = get_official_docker_image(instance_id)
        assert "swebench" in image
        assert "sweb.eval.x86_64" in image
        assert "django" in image.lower()
        assert "12345" in image

    def test_extract_custom_tag_from_official_image(self):
        """Verify extract_custom_tag works correctly with SWE-bench images."""
        from benchmarks.swtbench.run_infer import extract_custom_tag

        # Test extracting custom tag from official SWE-bench image format
        base_image = (
            "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest"
        )
        custom_tag = extract_custom_tag(base_image)
        assert custom_tag == "sweb.eval.x86_64.django_1776_django-12155"

    def test_agent_server_image_tag_format(self):
        """Verify the agent server image tag format matches expected pattern."""
        from benchmarks.swtbench.run_infer import (
            extract_custom_tag,
            get_official_docker_image,
        )
        from benchmarks.utils.version import SDK_SHORT_SHA

        instance_id = "astropy__astropy-13977"
        official_image = get_official_docker_image(instance_id)
        custom_tag = extract_custom_tag(official_image)

        # For source-minimal target
        suffix = f"-{DEFAULT_BUILD_TARGET}"
        expected_tag = f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"

        # The tag should follow the format:
        # ghcr.io/openhands/eval-agent-server:<SDK_SHA>-sweb.eval.x86_64.<repo>_1776_<name>-source-minimal
        assert expected_tag.startswith("ghcr.io/openhands/eval-agent-server:")
        assert SDK_SHORT_SHA in expected_tag
        assert "sweb.eval.x86_64" in expected_tag
        assert "source-minimal" in expected_tag


class TestSWTBenchPrepareWorkspace:
    """Tests for SWT-bench workspace preparation."""

    @pytest.fixture
    def mock_metadata(self):
        """Create mock EvalMetadata for tests."""
        from benchmarks.utils.critics import AgentFinishedCritic
        from openhands.sdk import LLM

        llm = LLM(model="test-model", api_key="test-key")

        return EvalMetadata(
            llm=llm,
            dataset="eth-sri/SWT-bench_Verified_bm25_27k_zsp",
            dataset_split="test",
            max_iterations=10,
            eval_output_dir="/tmp/test_output",
            details={},
            prompt_path="/tmp/prompt.j2",
            workspace_type="docker",
            critic=AgentFinishedCritic(),
        )

    @pytest.fixture
    def mock_instance(self):
        """Create mock EvalInstance for tests."""
        return EvalInstance(
            id="astropy__astropy-13977",
            data={
                "instance_id": "astropy__astropy-13977",
                "repo": "astropy/astropy",
                "base_commit": "abc123",
            },
        )

    @patch("benchmarks.swtbench.run_infer.build_image")
    @patch("benchmarks.swtbench.run_infer.DockerWorkspace")
    def test_prepare_workspace_calls_build_image_when_skip_build_false(
        self, mock_docker_workspace, mock_build_image, mock_metadata, mock_instance
    ):
        """Verify prepare_workspace calls build_image when SKIP_BUILD=0."""
        # Set SKIP_BUILD=0
        with patch.dict(os.environ, {"SKIP_BUILD": "0"}):
            # Mock build_image to return successful output
            from benchmarks.utils.build_utils import BuildOutput
            from benchmarks.utils.version import SDK_SHORT_SHA

            expected_tag = f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-sweb.eval.x86_64.astropy_1776_astropy-13977-source-minimal"
            mock_build_image.return_value = BuildOutput(
                base_image="docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-13977:latest",
                tags=[expected_tag],
                error=None,
            )

            # Mock DockerWorkspace
            mock_workspace_instance = MagicMock()
            mock_workspace_instance.execute_command = MagicMock(
                return_value=MagicMock(exit_code=0, stderr="", stdout="")
            )
            mock_docker_workspace.return_value = mock_workspace_instance

            # Create evaluation and call prepare_workspace
            evaluation = SWTBenchEvaluation(metadata=mock_metadata, num_workers=1)
            _ = evaluation.prepare_workspace(mock_instance)

            # Verify build_image was called
            mock_build_image.assert_called_once()
            call_args = mock_build_image.call_args
            assert call_args.kwargs["push"] is False
            assert "astropy" in call_args.kwargs["base_image"].lower()

    @patch("benchmarks.swtbench.run_infer.DockerWorkspace")
    def test_prepare_workspace_skips_build_when_skip_build_true(
        self, mock_docker_workspace, mock_metadata, mock_instance
    ):
        """Verify prepare_workspace skips build_image when SKIP_BUILD=1."""
        # Set SKIP_BUILD=1 (default)
        with patch.dict(os.environ, {"SKIP_BUILD": "1"}):
            # Mock DockerWorkspace
            mock_workspace_instance = MagicMock()
            mock_workspace_instance.execute_command = MagicMock(
                return_value=MagicMock(exit_code=0, stderr="", stdout="")
            )
            mock_docker_workspace.return_value = mock_workspace_instance

            # Create evaluation and call prepare_workspace
            evaluation = SWTBenchEvaluation(metadata=mock_metadata, num_workers=1)

            # build_image should not be called since SKIP_BUILD=1
            with patch("benchmarks.swtbench.run_infer.build_image") as mock_build:
                _ = evaluation.prepare_workspace(mock_instance)
                mock_build.assert_not_called()

            # DockerWorkspace should still be created
            mock_docker_workspace.assert_called_once()


class TestSWTBenchNoDevWorkspace:
    """Tests verifying SWT-bench no longer uses DockerDevWorkspace."""

    def test_no_docker_dev_workspace_import(self):
        """Verify DockerDevWorkspace is not imported in SWT-bench run_infer."""
        import benchmarks.swtbench.run_infer as swtbench_module

        # Check that DockerDevWorkspace is not in the module's namespace
        assert not hasattr(swtbench_module, "DockerDevWorkspace")

        # Also verify it's not in the module's imports by checking __dict__
        module_names = dir(swtbench_module)
        assert "DockerDevWorkspace" not in module_names
