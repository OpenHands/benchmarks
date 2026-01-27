"""Tests for commit0 constants.py."""

from benchmarks.commit0.constants import (
    AGENT_BRANCH_NAME,
    BUILD_TARGET,
    CUSTOM_TAG_PREFIX,
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_CONVERSATION_TIMEOUT,
    DEFAULT_DATASET,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_DOCKER_IMAGE_PREFIX,
    DEFAULT_IMAGE_TAG,
    DEFAULT_MODEL_NAME,
    DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT,
    DEFAULT_REPO_SPLIT,
    DEFAULT_RUNTIME_API_URL,
    GIT_BRANCH_NAME,
    TOTAL_INSTANCES,
    WORKSPACE_DIR,
)


class TestDatasetConstants:
    """Tests for dataset-related constants."""

    def test_default_dataset_is_valid_huggingface_path(self):
        """Test that DEFAULT_DATASET follows HuggingFace dataset path format."""
        assert "/" in DEFAULT_DATASET
        assert DEFAULT_DATASET == "wentingzhao/commit0_combined"

    def test_default_dataset_split(self):
        """Test that DEFAULT_DATASET_SPLIT is a valid split name."""
        assert DEFAULT_DATASET_SPLIT == "test"

    def test_default_repo_split(self):
        """Test that DEFAULT_REPO_SPLIT is a valid repo split."""
        assert DEFAULT_REPO_SPLIT in ["lite", "all"]


class TestDockerConstants:
    """Tests for Docker-related constants."""

    def test_default_docker_image_prefix_format(self):
        """Test that DEFAULT_DOCKER_IMAGE_PREFIX is a valid Docker registry prefix."""
        assert DEFAULT_DOCKER_IMAGE_PREFIX.endswith("/")
        assert "docker.io" in DEFAULT_DOCKER_IMAGE_PREFIX

    def test_default_image_tag(self):
        """Test that DEFAULT_IMAGE_TAG is a valid tag format."""
        assert DEFAULT_IMAGE_TAG == "v0"
        assert not DEFAULT_IMAGE_TAG.startswith(":")

    def test_custom_tag_prefix(self):
        """Test that CUSTOM_TAG_PREFIX is a valid prefix."""
        assert CUSTOM_TAG_PREFIX == "commit0-"
        assert CUSTOM_TAG_PREFIX.endswith("-")


class TestBuildConstants:
    """Tests for build-related constants."""

    def test_build_target(self):
        """Test that BUILD_TARGET is a valid build target."""
        assert BUILD_TARGET == "source-minimal"


class TestWorkspaceConstants:
    """Tests for workspace-related constants."""

    def test_workspace_dir_is_absolute_path(self):
        """Test that WORKSPACE_DIR is an absolute path."""
        assert WORKSPACE_DIR.startswith("/")
        assert WORKSPACE_DIR == "/workspace"


class TestGitConstants:
    """Tests for Git-related constants."""

    def test_git_branch_name(self):
        """Test that GIT_BRANCH_NAME is a valid branch name."""
        assert GIT_BRANCH_NAME == "commit0_combined"
        assert " " not in GIT_BRANCH_NAME

    def test_agent_branch_name(self):
        """Test that AGENT_BRANCH_NAME is a valid branch name."""
        assert AGENT_BRANCH_NAME == "openhands"
        assert " " not in AGENT_BRANCH_NAME


class TestModelConstants:
    """Tests for model-related constants."""

    def test_default_model_name(self):
        """Test that DEFAULT_MODEL_NAME is set."""
        assert DEFAULT_MODEL_NAME == "openhands"


class TestRuntimeConstants:
    """Tests for runtime-related constants."""

    def test_default_runtime_api_url_is_valid_url(self):
        """Test that DEFAULT_RUNTIME_API_URL is a valid URL."""
        assert DEFAULT_RUNTIME_API_URL.startswith("https://")
        assert "runtime" in DEFAULT_RUNTIME_API_URL

    def test_default_remote_runtime_startup_timeout_is_positive(self):
        """Test that DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT is positive."""
        assert DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT > 0
        assert DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT == 600

    def test_default_conversation_timeout_is_positive(self):
        """Test that DEFAULT_CONVERSATION_TIMEOUT is positive."""
        assert DEFAULT_CONVERSATION_TIMEOUT > 0
        assert DEFAULT_CONVERSATION_TIMEOUT == 3600

    def test_default_command_timeout_is_positive(self):
        """Test that DEFAULT_COMMAND_TIMEOUT is positive."""
        assert DEFAULT_COMMAND_TIMEOUT > 0
        assert DEFAULT_COMMAND_TIMEOUT == 600


class TestEvaluationConstants:
    """Tests for evaluation-related constants."""

    def test_total_instances_is_positive(self):
        """Test that TOTAL_INSTANCES is positive."""
        assert TOTAL_INSTANCES > 0
        assert TOTAL_INSTANCES == 16


class TestConstantsIntegration:
    """Integration tests for constants usage."""

    def test_docker_image_can_be_constructed(self):
        """Test that a valid Docker image name can be constructed from constants."""
        repo_name = "test-repo"
        image = f"{DEFAULT_DOCKER_IMAGE_PREFIX}{repo_name}:{DEFAULT_IMAGE_TAG}"
        assert image == "docker.io/wentingzhao/test-repo:v0"

    def test_custom_tag_can_be_constructed(self):
        """Test that a valid custom tag can be constructed from constants."""
        repo_name = "test-repo"
        custom_tag = f"{CUSTOM_TAG_PREFIX}{repo_name}"
        assert custom_tag == "commit0-test-repo"

    def test_workspace_path_can_be_constructed(self):
        """Test that a valid workspace path can be constructed from constants."""
        repo_name = "test-repo"
        workspace_path = f"{WORKSPACE_DIR}/{repo_name}"
        assert workspace_path == "/workspace/test-repo"

    def test_clone_command_can_be_constructed(self):
        """Test that a valid git clone command can be constructed from constants."""
        repo = "owner/test-repo"
        repo_name = repo.split("/")[1]
        clone_cmd = f"cd {WORKSPACE_DIR}/ && git clone -b {GIT_BRANCH_NAME} https://github.com/{repo}.git {repo_name}"
        assert (
            clone_cmd
            == "cd /workspace/ && git clone -b commit0_combined https://github.com/owner/test-repo.git test-repo"
        )
