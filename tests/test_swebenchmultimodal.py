"""Tests for SWE-Bench Multimodal eval_infer functionality."""

import tempfile

from benchmarks.swebenchmultimodal import constants
from benchmarks.swebenchmultimodal.eval_infer import convert_to_swebench_format


class TestConstants:
    """Tests for SWE-Bench Multimodal constants."""

    def test_default_dataset_is_string(self):
        """Test that DEFAULT_DATASET is a non-empty string."""
        assert isinstance(constants.DEFAULT_DATASET, str)
        assert len(constants.DEFAULT_DATASET) > 0
        assert "SWE-bench_Multimodal" in constants.DEFAULT_DATASET

    def test_default_split_is_string(self):
        """Test that DEFAULT_SPLIT is a non-empty string."""
        assert isinstance(constants.DEFAULT_SPLIT, str)
        assert len(constants.DEFAULT_SPLIT) > 0

    def test_docker_image_prefix_is_string(self):
        """Test that DOCKER_IMAGE_PREFIX is a valid docker prefix."""
        assert isinstance(constants.DOCKER_IMAGE_PREFIX, str)
        assert constants.DOCKER_IMAGE_PREFIX.endswith("/")

    def test_build_target_is_string(self):
        """Test that BUILD_TARGET is a non-empty string."""
        assert isinstance(constants.BUILD_TARGET, str)
        assert len(constants.BUILD_TARGET) > 0

    def test_workspace_dir_is_string(self):
        """Test that WORKSPACE_DIR is a valid path string."""
        assert isinstance(constants.WORKSPACE_DIR, str)
        assert constants.WORKSPACE_DIR.startswith("/")

    def test_env_variable_names_are_strings(self):
        """Test that environment variable names are non-empty strings."""
        env_vars = [
            constants.ENV_SKIP_BUILD,
            constants.ENV_RUNTIME_API_KEY,
            constants.ENV_SDK_SHORT_SHA,
            constants.ENV_REMOTE_RUNTIME_STARTUP_TIMEOUT,
            constants.ENV_RUNTIME_API_URL,
        ]
        for env_var in env_vars:
            assert isinstance(env_var, str)
            assert len(env_var) > 0

    def test_default_env_values_are_strings(self):
        """Test that default environment values are non-empty strings."""
        defaults = [
            constants.DEFAULT_SKIP_BUILD,
            constants.DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT,
            constants.DEFAULT_RUNTIME_API_URL,
        ]
        for default in defaults:
            assert isinstance(default, str)
            assert len(default) > 0

    def test_git_config_values_are_strings(self):
        """Test that git configuration values are non-empty strings."""
        git_values = [
            constants.GIT_USER_EMAIL,
            constants.GIT_USER_NAME,
            constants.GIT_COMMIT_MESSAGE,
        ]
        for value in git_values:
            assert isinstance(value, str)
            assert len(value) > 0

    def test_env_setup_commands_is_list(self):
        """Test that ENV_SETUP_COMMANDS is a non-empty list of strings."""
        assert isinstance(constants.ENV_SETUP_COMMANDS, list)
        assert len(constants.ENV_SETUP_COMMANDS) > 0
        for cmd in constants.ENV_SETUP_COMMANDS:
            assert isinstance(cmd, str)

    def test_allowed_image_types_is_list(self):
        """Test that ALLOWED_IMAGE_TYPES is a non-empty list of MIME types."""
        assert isinstance(constants.ALLOWED_IMAGE_TYPES, list)
        assert len(constants.ALLOWED_IMAGE_TYPES) > 0
        for mime_type in constants.ALLOWED_IMAGE_TYPES:
            assert isinstance(mime_type, str)
            assert mime_type.startswith("image/")

    def test_eval_workers_is_numeric_string(self):
        """Test that DEFAULT_EVAL_WORKERS is a numeric string."""
        assert isinstance(constants.DEFAULT_EVAL_WORKERS, str)
        assert constants.DEFAULT_EVAL_WORKERS.isdigit()

    def test_default_model_name_is_string(self):
        """Test that DEFAULT_MODEL_NAME is a non-empty string."""
        assert isinstance(constants.DEFAULT_MODEL_NAME, str)
        assert len(constants.DEFAULT_MODEL_NAME) > 0

    def test_solveable_keyword_is_string(self):
        """Test that SOLVEABLE_KEYWORD is a non-empty string."""
        assert isinstance(constants.SOLVEABLE_KEYWORD, str)
        assert len(constants.SOLVEABLE_KEYWORD) > 0

    def test_setup_files_to_remove_is_list(self):
        """Test that SETUP_FILES_TO_REMOVE is a non-empty list of filenames."""
        assert isinstance(constants.SETUP_FILES_TO_REMOVE, list)
        assert len(constants.SETUP_FILES_TO_REMOVE) > 0
        for filename in constants.SETUP_FILES_TO_REMOVE:
            assert isinstance(filename, str)
            assert len(filename) > 0

    def test_annotations_filename_is_string(self):
        """Test that ANNOTATIONS_FILENAME is a non-empty string."""
        assert isinstance(constants.ANNOTATIONS_FILENAME, str)
        assert len(constants.ANNOTATIONS_FILENAME) > 0
        assert constants.ANNOTATIONS_FILENAME.endswith(".json")


class TestConvertToSwebenchFormat:
    """Tests for convert_to_swebench_format function."""

    def test_empty_input_file_does_not_raise(self):
        """Test that an empty input file does not raise an exception."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as infile:
            infile.write("")  # Empty file
            input_path = infile.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".swebench.jsonl", delete=False
        ) as outfile:
            output_path = outfile.name

        # Should not raise - let the harness handle empty results
        convert_to_swebench_format(input_path, output_path)

        # Verify output file is empty
        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 0
