"""Tests for SWT-bench run_infer module.

Tests verify that SWT-bench uses correct image tagging and that Docker
workspace mode auto-builds images when they are missing locally (via
create_docker_workspace from image_utils).
"""

from unittest.mock import MagicMock, patch

import pytest

from benchmarks.swtbench.run_infer import (
    SWTBenchEvaluation,
    get_official_docker_image,
)
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.models import EvalInstance, EvalMetadata


class TestSWTBenchImageTagGeneration:
    """Tests for SWT-bench Docker image tag generation."""

    def test_swtbench_get_official_docker_image(self):
        """Verify get_official_docker_image produces correct format."""
        instance_id = "django__django-12345"
        image = get_official_docker_image(instance_id)
        assert "swebench" in image
        assert "sweb.eval.x86_64" in image
        assert "django" in image.lower()
        assert "12345" in image

    def test_agent_server_image_tag_format(self):
        """Verify the agent server image tag format matches expected pattern."""
        from benchmarks.utils.version import IMAGE_TAG_PREFIX

        instance_id = "astropy__astropy-13977"
        official_image = get_official_docker_image(instance_id)

        # Extract custom tag the same way prepare_workspace does
        name_tag = official_image.split("/")[-1]
        custom_tag = name_tag.split(":")[0]
        build_target = "source-minimal"
        suffix = f"-{build_target}"

        expected_tag = f"{EVAL_AGENT_SERVER_IMAGE}:{IMAGE_TAG_PREFIX}-{custom_tag}{suffix}"

        # The tag should follow the format:
        # ghcr.io/openhands/eval-agent-server:<IMAGE_TAG_PREFIX>-sweb.eval.x86_64.<repo>_1776_<name>-source-minimal
        assert expected_tag.startswith("ghcr.io/openhands/eval-agent-server:")
        assert IMAGE_TAG_PREFIX in expected_tag
        assert "sweb.eval.x86_64" in expected_tag
        assert "source-minimal" in expected_tag


class TestLocalDockerImageExists:
    """Tests for local_image_exists helper in image_utils."""

    @patch("benchmarks.utils.image_utils.subprocess.run")
    def test_returns_true_when_image_exists(self, mock_run):
        from benchmarks.utils.image_utils import local_image_exists

        mock_run.return_value = MagicMock(returncode=0)
        assert local_image_exists("some-image:tag") is True
        mock_run.assert_called_once_with(
            ["docker", "image", "inspect", "some-image:tag"],
            capture_output=True,
            check=False,
            timeout=5,
        )

    @patch("benchmarks.utils.image_utils.subprocess.run")
    def test_returns_false_when_image_missing(self, mock_run):
        from benchmarks.utils.image_utils import local_image_exists

        mock_run.return_value = MagicMock(returncode=1)
        assert local_image_exists("missing-image:tag") is False


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

    @patch("benchmarks.swtbench.run_infer.create_docker_workspace")
    def test_prepare_workspace_delegates_to_create_docker_workspace(
        self, mock_create_workspace, mock_metadata, mock_instance
    ):
        """Verify prepare_workspace delegates to create_docker_workspace for docker."""
        mock_workspace_instance = MagicMock()
        mock_workspace_instance.execute_command = MagicMock(
            return_value=MagicMock(exit_code=0, stderr="", stdout="")
        )
        mock_create_workspace.return_value = mock_workspace_instance

        evaluation = SWTBenchEvaluation(metadata=mock_metadata, num_workers=1)
        _ = evaluation.prepare_workspace(mock_instance)

        mock_create_workspace.assert_called_once()
        call_args = mock_create_workspace.call_args
        assert "astropy" in call_args.kwargs["base_image"].lower()
        assert call_args.kwargs["build_target"] == "source-minimal"

    @patch("benchmarks.utils.image_utils.local_image_exists", return_value=True)
    @patch("benchmarks.swtbench.run_infer.create_docker_workspace")
    def test_uses_prebuilt_image_when_exists_locally(
        self,
        mock_create_workspace,
        mock_local_exists,
        mock_metadata,
        mock_instance,
    ):
        """When image exists locally, create_docker_workspace returns DockerWorkspace."""
        from openhands.workspace import DockerWorkspace

        mock_workspace_instance = MagicMock(spec=DockerWorkspace)
        mock_workspace_instance.execute_command = MagicMock(
            return_value=MagicMock(exit_code=0, stderr="", stdout="")
        )
        mock_create_workspace.return_value = mock_workspace_instance

        evaluation = SWTBenchEvaluation(metadata=mock_metadata, num_workers=1)
        workspace = evaluation.prepare_workspace(mock_instance)

        mock_create_workspace.assert_called_once()
        assert workspace is mock_workspace_instance

    @patch("benchmarks.utils.image_utils.local_image_exists", return_value=False)
    @patch("benchmarks.swtbench.run_infer.create_docker_workspace")
    def test_builds_when_image_missing_locally(
        self,
        mock_create_workspace,
        mock_local_exists,
        mock_metadata,
        mock_instance,
    ):
        """When image is missing locally, create_docker_workspace builds it."""
        from openhands.workspace import DockerDevWorkspace

        mock_workspace_instance = MagicMock(spec=DockerDevWorkspace)
        mock_workspace_instance.execute_command = MagicMock(
            return_value=MagicMock(exit_code=0, stderr="", stdout="")
        )
        mock_create_workspace.return_value = mock_workspace_instance

        evaluation = SWTBenchEvaluation(metadata=mock_metadata, num_workers=1)
        workspace = evaluation.prepare_workspace(mock_instance)

        mock_create_workspace.assert_called_once()
        assert workspace is mock_workspace_instance


class TestSWTBenchNoDevWorkspace:
    """Tests verifying SWT-bench run_infer does not directly import DockerDevWorkspace."""

    def test_no_docker_dev_workspace_import(self):
        """Verify DockerDevWorkspace is not imported in SWT-bench run_infer."""
        import benchmarks.swtbench.run_infer as swtbench_module

        # Check that DockerDevWorkspace is not in the module's namespace
        assert not hasattr(swtbench_module, "DockerDevWorkspace")

        # Also verify it's not in the module's imports by checking __dict__
        module_names = dir(swtbench_module)
        assert "DockerDevWorkspace" not in module_names
