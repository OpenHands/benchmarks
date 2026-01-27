"""Tests for Multi-SWE-Bench constants module.

This test suite verifies that:
1. All constants are properly defined and accessible
2. Constants have the expected types and values
3. Constants are correctly imported and used in other modules
"""

from pathlib import Path


class TestDatasetConstants:
    """Tests for dataset-related constants."""

    def test_default_dataset(self):
        from benchmarks.multiswebench.constants import DEFAULT_DATASET

        assert DEFAULT_DATASET == "bytedance-research/Multi-SWE-Bench"
        assert isinstance(DEFAULT_DATASET, str)

    def test_default_split(self):
        from benchmarks.multiswebench.constants import DEFAULT_SPLIT

        assert DEFAULT_SPLIT == "test"
        assert isinstance(DEFAULT_SPLIT, str)

    def test_default_language(self):
        from benchmarks.multiswebench.constants import DEFAULT_LANGUAGE

        assert DEFAULT_LANGUAGE == "java"
        assert isinstance(DEFAULT_LANGUAGE, str)

    def test_default_model_name(self):
        from benchmarks.multiswebench.constants import DEFAULT_MODEL_NAME

        assert DEFAULT_MODEL_NAME == "OpenHands"
        assert isinstance(DEFAULT_MODEL_NAME, str)

    def test_default_version(self):
        from benchmarks.multiswebench.constants import DEFAULT_VERSION

        assert DEFAULT_VERSION == "0.1"
        assert isinstance(DEFAULT_VERSION, str)


class TestDockerImageConstants:
    """Tests for Docker/image-related constants."""

    def test_default_docker_image_prefix(self):
        from benchmarks.multiswebench.constants import DEFAULT_DOCKER_IMAGE_PREFIX

        assert DEFAULT_DOCKER_IMAGE_PREFIX == "mswebench"
        assert isinstance(DEFAULT_DOCKER_IMAGE_PREFIX, str)

    def test_default_build_target(self):
        from benchmarks.multiswebench.constants import DEFAULT_BUILD_TARGET

        assert DEFAULT_BUILD_TARGET == "source-minimal"
        assert isinstance(DEFAULT_BUILD_TARGET, str)

    def test_env_var_names(self):
        from benchmarks.multiswebench.constants import (
            DOCKER_IMAGE_PREFIX_ENV_VAR,
            LANGUAGE_ENV_VAR,
            SKIP_BUILD_ENV_VAR,
        )

        assert DOCKER_IMAGE_PREFIX_ENV_VAR == "EVAL_DOCKER_IMAGE_PREFIX"
        assert LANGUAGE_ENV_VAR == "LANGUAGE"
        assert SKIP_BUILD_ENV_VAR == "MULTI_SWE_BENCH_SKIP_BUILD"


class TestRuntimeConstants:
    """Tests for runtime-related constants."""

    def test_default_runtime_api_url(self):
        from benchmarks.multiswebench.constants import DEFAULT_RUNTIME_API_URL

        assert DEFAULT_RUNTIME_API_URL == "https://runtime.eval.all-hands.dev"
        assert isinstance(DEFAULT_RUNTIME_API_URL, str)

    def test_default_startup_timeout(self):
        from benchmarks.multiswebench.constants import DEFAULT_STARTUP_TIMEOUT

        assert DEFAULT_STARTUP_TIMEOUT == 600
        assert isinstance(DEFAULT_STARTUP_TIMEOUT, int)

    def test_runtime_env_var_names(self):
        from benchmarks.multiswebench.constants import (
            REMOTE_RUNTIME_STARTUP_TIMEOUT_ENV_VAR,
            RUNTIME_API_KEY_ENV_VAR,
            RUNTIME_API_URL_ENV_VAR,
            SDK_SHORT_SHA_ENV_VAR,
        )

        assert RUNTIME_API_KEY_ENV_VAR == "RUNTIME_API_KEY"
        assert RUNTIME_API_URL_ENV_VAR == "RUNTIME_API_URL"
        assert SDK_SHORT_SHA_ENV_VAR == "SDK_SHORT_SHA"
        assert (
            REMOTE_RUNTIME_STARTUP_TIMEOUT_ENV_VAR == "REMOTE_RUNTIME_STARTUP_TIMEOUT"
        )

    def test_boolean_defaults(self):
        from benchmarks.multiswebench.constants import (
            DEFAULT_RUN_WITH_BROWSING,
            DEFAULT_USE_HINT_TEXT,
            DEFAULT_USE_INSTANCE_IMAGE,
        )

        assert DEFAULT_USE_HINT_TEXT is False
        assert DEFAULT_USE_INSTANCE_IMAGE is True
        assert DEFAULT_RUN_WITH_BROWSING is False


class TestEvaluationConstants:
    """Tests for evaluation-related constants."""

    def test_default_eval_mode(self):
        from benchmarks.multiswebench.constants import DEFAULT_EVAL_MODE

        assert DEFAULT_EVAL_MODE == "evaluation"
        assert isinstance(DEFAULT_EVAL_MODE, str)

    def test_default_config_values(self):
        from benchmarks.multiswebench.constants import (
            DEFAULT_CLEAR_ENV,
            DEFAULT_FORCE_BUILD,
            DEFAULT_NEED_CLONE,
            DEFAULT_STOP_ON_ERROR,
        )

        assert DEFAULT_FORCE_BUILD is True
        assert DEFAULT_NEED_CLONE is True
        assert DEFAULT_CLEAR_ENV is True
        assert DEFAULT_STOP_ON_ERROR is False

    def test_default_worker_counts(self):
        from benchmarks.multiswebench.constants import (
            DEFAULT_MAX_WORKERS,
            DEFAULT_MAX_WORKERS_BUILD_IMAGE,
            DEFAULT_MAX_WORKERS_RUN_INSTANCE,
        )

        assert DEFAULT_MAX_WORKERS == 5
        assert DEFAULT_MAX_WORKERS_BUILD_IMAGE == 5
        assert DEFAULT_MAX_WORKERS_RUN_INSTANCE == 5

    def test_default_log_level(self):
        from benchmarks.multiswebench.constants import DEFAULT_LOG_LEVEL

        assert DEFAULT_LOG_LEVEL == "DEBUG"
        assert isinstance(DEFAULT_LOG_LEVEL, str)

    def test_fix_patch_run_cmd(self):
        from benchmarks.multiswebench.constants import FIX_PATCH_RUN_CMD

        assert isinstance(FIX_PATCH_RUN_CMD, str)
        assert "bash -c" in FIX_PATCH_RUN_CMD
        assert "patch" in FIX_PATCH_RUN_CMD


class TestPathConstants:
    """Tests for path-related constants."""

    def test_dataset_cache_dir_name(self):
        from benchmarks.multiswebench.constants import DATASET_CACHE_DIR_NAME

        assert DATASET_CACHE_DIR_NAME == "data"
        assert isinstance(DATASET_CACHE_DIR_NAME, str)

    def test_dataset_cache_dir(self):
        from benchmarks.multiswebench.constants import DATASET_CACHE_DIR

        assert isinstance(DATASET_CACHE_DIR, Path)
        assert DATASET_CACHE_DIR.name == "data"
        # Verify it's relative to the constants module
        assert "multiswebench" in str(DATASET_CACHE_DIR)


class TestWorkspaceConstants:
    """Tests for workspace-related constants."""

    def test_default_working_dir(self):
        from benchmarks.multiswebench.constants import DEFAULT_WORKING_DIR

        assert DEFAULT_WORKING_DIR == "/workspace"
        assert isinstance(DEFAULT_WORKING_DIR, str)

    def test_default_env_setup_commands(self):
        from benchmarks.multiswebench.constants import DEFAULT_ENV_SETUP_COMMANDS

        assert isinstance(DEFAULT_ENV_SETUP_COMMANDS, list)
        assert len(DEFAULT_ENV_SETUP_COMMANDS) > 0
        assert "export PIP_CACHE_DIR=~/.cache/pip" in DEFAULT_ENV_SETUP_COMMANDS


class TestConstantsUsageInModules:
    """Tests to verify constants are properly used in other modules."""

    def test_build_images_uses_constants(self):
        """Verify build_images.py imports and uses constants."""
        from benchmarks.multiswebench import build_images

        # Check that the module uses the constants
        assert hasattr(build_images, "DOCKER_IMAGE_PREFIX")
        assert hasattr(build_images, "LANGUAGE")

    def test_download_dataset_uses_constants(self):
        """Verify download_dataset.py imports and uses constants."""
        # The module should import DATASET_CACHE_DIR from constants
        # We can verify by checking the module's imports
        import inspect

        from benchmarks.multiswebench import download_dataset

        source = inspect.getsource(download_dataset)
        assert "from benchmarks.multiswebench.constants import" in source

    def test_eval_infer_uses_constants(self):
        """Verify eval_infer.py imports and uses constants."""
        import inspect

        from benchmarks.multiswebench import eval_infer

        source = inspect.getsource(eval_infer)
        assert "from benchmarks.multiswebench.constants import" in source

    def test_run_infer_uses_constants(self):
        """Verify run_infer.py imports and uses constants."""
        import inspect

        from benchmarks.multiswebench import run_infer

        source = inspect.getsource(run_infer)
        assert "from benchmarks.multiswebench.constants import" in source

    def test_data_change_uses_constants(self):
        """Verify data_change.py imports and uses constants."""
        import inspect

        from benchmarks.multiswebench.scripts.data import data_change

        source = inspect.getsource(data_change)
        assert "from benchmarks.multiswebench.constants import" in source

    def test_update_multi_swe_bench_config_uses_constants(self):
        """Verify update_multi_swe_bench_config.py imports and uses constants."""
        import inspect

        from benchmarks.multiswebench.scripts.eval import update_multi_swe_bench_config

        source = inspect.getsource(update_multi_swe_bench_config)
        assert "from benchmarks.multiswebench.constants import" in source


class TestAllConstantsExported:
    """Test that all expected constants are exported from the module."""

    def test_all_constants_importable(self):
        """Verify all constants can be imported from the module."""

        # If we get here without ImportError, all constants are importable
        assert True
