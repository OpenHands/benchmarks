"""Tests for constants module.

This test suite verifies that:
1. Shared constants are properly defined in benchmarks/utils/constants.py
2. Benchmark-specific constants are properly defined in {benchmark}/constants.py
3. All constants can be imported correctly
"""


class TestSharedConstants:
    """Test shared constants in benchmarks/utils/constants.py."""

    def test_shared_constants_import(self):
        """Test that shared constants can be imported."""
        from benchmarks.utils.constants import (
            DEFAULT_CRITIC,
            DEFAULT_DATASET,
            DEFAULT_ENV_SETUP_COMMANDS,
            DEFAULT_EVAL_LIMIT,
            DEFAULT_MAX_ATTEMPTS,
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_MAX_RETRIES,
            DEFAULT_NUM_WORKERS,
            DEFAULT_OUTPUT_DIR,
            DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT,
            DEFAULT_RUNTIME_API_URL,
            DEFAULT_SPLIT,
            DEFAULT_WORKSPACE_TYPE,
            EVAL_AGENT_SERVER_IMAGE,
            OUTPUT_FILENAME,
        )

        # Verify values are not None
        assert OUTPUT_FILENAME is not None
        assert EVAL_AGENT_SERVER_IMAGE is not None
        assert DEFAULT_DATASET is not None
        assert DEFAULT_SPLIT is not None
        assert DEFAULT_MAX_ITERATIONS is not None
        assert DEFAULT_MAX_ATTEMPTS is not None
        assert DEFAULT_MAX_RETRIES is not None
        assert DEFAULT_NUM_WORKERS is not None
        assert DEFAULT_EVAL_LIMIT is not None
        assert DEFAULT_WORKSPACE_TYPE is not None
        assert DEFAULT_OUTPUT_DIR is not None
        assert DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT is not None
        assert DEFAULT_RUNTIME_API_URL is not None
        assert DEFAULT_ENV_SETUP_COMMANDS is not None
        assert DEFAULT_CRITIC is not None

    def test_shared_constants_values(self):
        """Test that shared constants have expected values."""
        from benchmarks.utils.constants import (
            DEFAULT_DATASET,
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT,
            DEFAULT_RUNTIME_API_URL,
            DEFAULT_SPLIT,
            DEFAULT_WORKSPACE_TYPE,
            EVAL_AGENT_SERVER_IMAGE,
            OUTPUT_FILENAME,
        )

        assert OUTPUT_FILENAME == "output.jsonl"
        assert EVAL_AGENT_SERVER_IMAGE == "ghcr.io/openhands/eval-agent-server"
        assert DEFAULT_DATASET == "princeton-nlp/SWE-bench_Verified"
        assert DEFAULT_SPLIT == "test"
        assert DEFAULT_MAX_ITERATIONS == 100
        assert DEFAULT_WORKSPACE_TYPE == "docker"
        assert DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT == 600
        assert "runtime.eval.all-hands.dev" in DEFAULT_RUNTIME_API_URL


class TestGAIAConstants:
    """Test GAIA benchmark-specific constants."""

    def test_gaia_constants_import(self):
        """Test that GAIA constants can be imported."""
        from benchmarks.gaia.constants import (
            DATASET_CACHE_DIR,
            GAIA_BASE_IMAGE,
            GAIA_DATA_YEAR,
            GAIA_DATASET,
            GAIA_DEFAULT_SPLIT,
        )

        assert GAIA_DATASET is not None
        assert GAIA_DEFAULT_SPLIT is not None
        assert GAIA_BASE_IMAGE is not None
        assert DATASET_CACHE_DIR is not None
        assert GAIA_DATA_YEAR is not None

    def test_gaia_constants_values(self):
        """Test that GAIA constants have expected values."""
        from benchmarks.gaia.constants import (
            GAIA_BASE_IMAGE,
            GAIA_DATA_YEAR,
            GAIA_DATASET,
            GAIA_DEFAULT_SPLIT,
        )

        assert GAIA_DATASET == "gaia-benchmark/GAIA"
        assert GAIA_DEFAULT_SPLIT == "validation"
        assert "python" in GAIA_BASE_IMAGE.lower()
        assert GAIA_DATA_YEAR == "2023"


class TestSWEBenchConstants:
    """Test SWE-bench benchmark-specific constants."""

    def test_swebench_constants_import(self):
        """Test that SWE-bench constants can be imported."""
        from benchmarks.swebench.constants import (
            DEFAULT_BUILD_TARGET,
            SWEBENCH_DATASET,
            SWEBENCH_DEFAULT_SPLIT,
            SWEBENCH_DOCKER_IMAGE_PREFIX,
            WRAPPED_REPOS,
        )

        assert SWEBENCH_DATASET is not None
        assert SWEBENCH_DEFAULT_SPLIT is not None
        assert SWEBENCH_DOCKER_IMAGE_PREFIX is not None
        assert WRAPPED_REPOS is not None
        assert DEFAULT_BUILD_TARGET is not None

    def test_swebench_constants_values(self):
        """Test that SWE-bench constants have expected values."""
        from benchmarks.swebench.constants import (
            DEFAULT_BUILD_TARGET,
            SWEBENCH_DATASET,
            SWEBENCH_DEFAULT_SPLIT,
            SWEBENCH_DOCKER_IMAGE_PREFIX,
            WRAPPED_REPOS,
        )

        assert SWEBENCH_DATASET == "princeton-nlp/SWE-bench_Verified"
        assert SWEBENCH_DEFAULT_SPLIT == "test"
        assert "swebench" in SWEBENCH_DOCKER_IMAGE_PREFIX.lower()
        assert isinstance(WRAPPED_REPOS, set)
        assert DEFAULT_BUILD_TARGET == "source-minimal"


class TestSWEBenchMultimodalConstants:
    """Test SWE-bench Multimodal benchmark-specific constants."""

    def test_swebenchmultimodal_constants_import(self):
        """Test that SWE-bench Multimodal constants can be imported."""
        from benchmarks.swebenchmultimodal.constants import (
            DEFAULT_BUILD_TARGET,
            SWEBENCH_MULTIMODAL_DATASET,
            SWEBENCH_MULTIMODAL_DEFAULT_SPLIT,
        )

        assert SWEBENCH_MULTIMODAL_DATASET is not None
        assert SWEBENCH_MULTIMODAL_DEFAULT_SPLIT is not None
        assert DEFAULT_BUILD_TARGET is not None

    def test_swebenchmultimodal_constants_values(self):
        """Test that SWE-bench Multimodal constants have expected values."""
        from benchmarks.swebenchmultimodal.constants import (
            DEFAULT_BUILD_TARGET,
            SWEBENCH_MULTIMODAL_DATASET,
            SWEBENCH_MULTIMODAL_DEFAULT_SPLIT,
        )

        assert SWEBENCH_MULTIMODAL_DATASET == "princeton-nlp/SWE-bench_Multimodal"
        assert SWEBENCH_MULTIMODAL_DEFAULT_SPLIT == "dev"
        assert DEFAULT_BUILD_TARGET == "source-minimal"


class TestMultiSWEBenchConstants:
    """Test Multi-SWE-bench benchmark-specific constants."""

    def test_multiswebench_constants_import(self):
        """Test that Multi-SWE-bench constants can be imported."""
        from benchmarks.multiswebench.constants import (
            DATASET_CACHE_DIR,
            DEFAULT_BUILD_TARGET,
            DEFAULT_DOCKER_IMAGE_PREFIX,
            DEFAULT_LANG,
            DOCKER_IMAGE_PREFIX,
            MULTISWEBENCH_DATASET,
            MULTISWEBENCH_DEFAULT_SPLIT,
            RUN_WITH_BROWSING,
            USE_HINT_TEXT,
            USE_INSTANCE_IMAGE,
        )

        assert MULTISWEBENCH_DATASET is not None
        assert MULTISWEBENCH_DEFAULT_SPLIT is not None
        assert DEFAULT_LANG is not None
        assert DEFAULT_DOCKER_IMAGE_PREFIX is not None
        assert DOCKER_IMAGE_PREFIX is not None
        assert DATASET_CACHE_DIR is not None
        assert DEFAULT_BUILD_TARGET is not None
        assert isinstance(USE_HINT_TEXT, bool)
        assert isinstance(USE_INSTANCE_IMAGE, bool)
        assert isinstance(RUN_WITH_BROWSING, bool)

    def test_multiswebench_constants_values(self):
        """Test that Multi-SWE-bench constants have expected values."""
        from benchmarks.multiswebench.constants import (
            DEFAULT_BUILD_TARGET,
            DEFAULT_DOCKER_IMAGE_PREFIX,
            DEFAULT_LANG,
            MULTISWEBENCH_DATASET,
            MULTISWEBENCH_DEFAULT_SPLIT,
        )

        assert MULTISWEBENCH_DATASET == "ByteDance-Seed/Multi-SWE-bench"
        assert MULTISWEBENCH_DEFAULT_SPLIT == "test"
        assert DEFAULT_LANG == "java"
        assert DEFAULT_DOCKER_IMAGE_PREFIX == "mswebench"
        assert DEFAULT_BUILD_TARGET == "source-minimal"


class TestCommit0Constants:
    """Test Commit0 benchmark-specific constants."""

    def test_commit0_constants_import(self):
        """Test that Commit0 constants can be imported."""
        from benchmarks.commit0.constants import (
            COMMIT0_DATASET,
            COMMIT0_DEFAULT_SPLIT,
            DEFAULT_BUILD_TARGET,
            DEFAULT_DOCKER_IMAGE_PREFIX,
            DEFAULT_REPO_SPLIT,
            DOCKER_IMAGE_PREFIX,
        )

        assert COMMIT0_DATASET is not None
        assert COMMIT0_DEFAULT_SPLIT is not None
        assert DEFAULT_REPO_SPLIT is not None
        assert DEFAULT_DOCKER_IMAGE_PREFIX is not None
        assert DOCKER_IMAGE_PREFIX is not None
        assert DEFAULT_BUILD_TARGET is not None

    def test_commit0_constants_values(self):
        """Test that Commit0 constants have expected values."""
        from benchmarks.commit0.constants import (
            COMMIT0_DATASET,
            COMMIT0_DEFAULT_SPLIT,
            DEFAULT_BUILD_TARGET,
            DEFAULT_DOCKER_IMAGE_PREFIX,
            DEFAULT_REPO_SPLIT,
        )

        assert COMMIT0_DATASET == "wentingzhao/commit0_combined"
        assert COMMIT0_DEFAULT_SPLIT == "test"
        assert DEFAULT_REPO_SPLIT == "lite"
        assert "wentingzhao" in DEFAULT_DOCKER_IMAGE_PREFIX
        assert DEFAULT_BUILD_TARGET == "source-minimal"


class TestSWTBenchConstants:
    """Test SWT-bench benchmark-specific constants."""

    def test_swtbench_constants_import(self):
        """Test that SWT-bench constants can be imported."""
        from benchmarks.swtbench.constants import (
            DEFAULT_BUILD_TARGET,
            SWTBENCH_DOCKER_IMAGE_PREFIX,
        )

        assert SWTBENCH_DOCKER_IMAGE_PREFIX is not None
        assert DEFAULT_BUILD_TARGET is not None

    def test_swtbench_constants_values(self):
        """Test that SWT-bench constants have expected values."""
        from benchmarks.swtbench.constants import (
            DEFAULT_BUILD_TARGET,
            SWTBENCH_DOCKER_IMAGE_PREFIX,
        )

        assert "swtbench" in SWTBENCH_DOCKER_IMAGE_PREFIX.lower()
        assert DEFAULT_BUILD_TARGET == "source-minimal"


class TestOpenAgentSafetyConstants:
    """Test OpenAgentSafety benchmark-specific constants."""

    def test_openagentsafety_constants_import(self):
        """Test that OpenAgentSafety constants can be imported."""
        from benchmarks.openagentsafety.constants import (
            DEFAULT_NPC_MODEL,
            OPENAGENTSAFETY_SERVER_IMAGE,
        )

        assert OPENAGENTSAFETY_SERVER_IMAGE is not None
        assert DEFAULT_NPC_MODEL is not None

    def test_openagentsafety_constants_values(self):
        """Test that OpenAgentSafety constants have expected values."""
        from benchmarks.openagentsafety.constants import (
            DEFAULT_NPC_MODEL,
            OPENAGENTSAFETY_SERVER_IMAGE,
        )

        assert "openagentsafety" in OPENAGENTSAFETY_SERVER_IMAGE.lower()
        assert "gpt" in DEFAULT_NPC_MODEL.lower()
