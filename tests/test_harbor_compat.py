"""Tests for Harbor benchmark compatibility mappings."""

import pytest

from benchmarks.terminalbench.config import INFER_DEFAULTS as TERMINAL_DEFAULTS
from benchmarks.utils.harbor_compat import (
    HARBOR_DATASET_BY_BENCHMARK,
    get_harbor_dataset,
    is_harbor_covered,
    normalize_benchmark_name,
)


def test_known_harbor_dataset_mappings() -> None:
    """Test Harbor dataset names for covered OpenHands benchmark modules."""
    assert get_harbor_dataset("swebench") == "swebench-verified"
    assert get_harbor_dataset("swebenchpro") == "swebenchpro"
    assert get_harbor_dataset("swebenchmultilingual") == "swebench_multilingual"
    assert get_harbor_dataset("swtbench") == "swtbench-verified"
    assert get_harbor_dataset("swesmith") == "swesmith"
    assert get_harbor_dataset("multiswebench") == "multi-swe-bench"
    assert get_harbor_dataset("gaia") == "gaia"
    assert get_harbor_dataset("terminalbench") == "terminal-bench@2.0"
    assert get_harbor_dataset("swegym") == "swegym"


def test_name_normalization_accepts_cli_style_names() -> None:
    """Test hyphen/underscore variants normalize to the same mapping key."""
    assert normalize_benchmark_name("SWE-Bench_Pro") == "swebenchpro"
    assert get_harbor_dataset("swe-bench-pro") == "swebenchpro"
    assert get_harbor_dataset("terminal-bench") == "terminal-bench@2.0"


def test_uncovered_benchmark_is_explicit() -> None:
    """Test benchmarks without Harbor adapters are not silently mapped."""
    assert is_harbor_covered("commit0") is False
    with pytest.raises(KeyError, match="No Harbor dataset mapping"):
        get_harbor_dataset("commit0")


def test_terminalbench_default_uses_mapping() -> None:
    """Test an existing Harbor-backed wrapper consumes the shared mapping."""
    assert TERMINAL_DEFAULTS["dataset"] == HARBOR_DATASET_BY_BENCHMARK["terminalbench"]
