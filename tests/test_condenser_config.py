"""Tests for condenser configuration in benchmarks."""

from benchmarks.commit0.config import (
    CONDENSER_DEFAULTS as COMMIT0_CONDENSER_DEFAULTS,
    INFER_DEFAULTS as COMMIT0_INFER_DEFAULTS,
)
from benchmarks.gaia.config import (
    CONDENSER_DEFAULTS as GAIA_CONDENSER_DEFAULTS,
    INFER_DEFAULTS as GAIA_INFER_DEFAULTS,
)
from benchmarks.swebench.config import CONDENSER_DEFAULTS, INFER_DEFAULTS
from benchmarks.swebenchmultimodal.config import (
    CONDENSER_DEFAULTS as SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS,
    INFER_DEFAULTS as SWEBENCHMULTIMODAL_INFER_DEFAULTS,
)
from benchmarks.swtbench.config import (
    CONDENSER_DEFAULTS as SWTBENCH_CONDENSER_DEFAULTS,
    INFER_DEFAULTS as SWTBENCH_INFER_DEFAULTS,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.critics import PassCritic
from benchmarks.utils.models import EvalMetadata
from openhands.sdk import LLM


def test_condenser_defaults_in_swebench_config():
    """Test that condenser defaults are properly defined in swebench config."""
    assert "enable_condenser" in CONDENSER_DEFAULTS
    assert "condenser_max_size" in CONDENSER_DEFAULTS
    assert "condenser_keep_first" in CONDENSER_DEFAULTS
    assert CONDENSER_DEFAULTS["enable_condenser"] is True
    assert CONDENSER_DEFAULTS["condenser_max_size"] == 240
    assert CONDENSER_DEFAULTS["condenser_keep_first"] == 2


def test_condenser_defaults_in_swtbench_config():
    """Test that condenser defaults are properly defined in swtbench config."""
    assert "enable_condenser" in SWTBENCH_CONDENSER_DEFAULTS
    assert "condenser_max_size" in SWTBENCH_CONDENSER_DEFAULTS
    assert "condenser_keep_first" in SWTBENCH_CONDENSER_DEFAULTS
    assert SWTBENCH_CONDENSER_DEFAULTS["enable_condenser"] is True
    assert SWTBENCH_CONDENSER_DEFAULTS["condenser_max_size"] == 240
    assert SWTBENCH_CONDENSER_DEFAULTS["condenser_keep_first"] == 2


def test_condenser_defaults_in_swebenchmultimodal_config():
    """Test that condenser defaults are properly defined in swebenchmultimodal config."""
    assert "enable_condenser" in SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS
    assert "condenser_max_size" in SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS
    assert "condenser_keep_first" in SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS
    assert SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS["enable_condenser"] is True
    assert SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS["condenser_max_size"] == 240
    assert SWEBENCHMULTIMODAL_CONDENSER_DEFAULTS["condenser_keep_first"] == 2


def test_condenser_defaults_in_gaia_config():
    """Test that condenser defaults are properly defined in gaia config."""
    assert "enable_condenser" in GAIA_CONDENSER_DEFAULTS
    assert "condenser_max_size" in GAIA_CONDENSER_DEFAULTS
    assert "condenser_keep_first" in GAIA_CONDENSER_DEFAULTS
    assert GAIA_CONDENSER_DEFAULTS["enable_condenser"] is True
    assert GAIA_CONDENSER_DEFAULTS["condenser_max_size"] == 240
    assert GAIA_CONDENSER_DEFAULTS["condenser_keep_first"] == 2


def test_condenser_defaults_in_commit0_config():
    """Test that condenser defaults are properly defined in commit0 config."""
    assert "enable_condenser" in COMMIT0_CONDENSER_DEFAULTS
    assert "condenser_max_size" in COMMIT0_CONDENSER_DEFAULTS
    assert "condenser_keep_first" in COMMIT0_CONDENSER_DEFAULTS
    assert COMMIT0_CONDENSER_DEFAULTS["enable_condenser"] is True
    assert COMMIT0_CONDENSER_DEFAULTS["condenser_max_size"] == 240
    assert COMMIT0_CONDENSER_DEFAULTS["condenser_keep_first"] == 2


def test_condenser_defaults_in_infer_defaults():
    """Test that condenser defaults are included in INFER_DEFAULTS."""
    assert "enable_condenser" in INFER_DEFAULTS
    assert "condenser_max_size" in INFER_DEFAULTS
    assert "condenser_keep_first" in INFER_DEFAULTS
    assert INFER_DEFAULTS["enable_condenser"] is True

    assert "enable_condenser" in SWTBENCH_INFER_DEFAULTS
    assert "condenser_max_size" in SWTBENCH_INFER_DEFAULTS
    assert "condenser_keep_first" in SWTBENCH_INFER_DEFAULTS
    assert SWTBENCH_INFER_DEFAULTS["enable_condenser"] is True

    assert "enable_condenser" in SWEBENCHMULTIMODAL_INFER_DEFAULTS
    assert "condenser_max_size" in SWEBENCHMULTIMODAL_INFER_DEFAULTS
    assert "condenser_keep_first" in SWEBENCHMULTIMODAL_INFER_DEFAULTS
    assert SWEBENCHMULTIMODAL_INFER_DEFAULTS["enable_condenser"] is True

    assert "enable_condenser" in GAIA_INFER_DEFAULTS
    assert "condenser_max_size" in GAIA_INFER_DEFAULTS
    assert "condenser_keep_first" in GAIA_INFER_DEFAULTS
    assert GAIA_INFER_DEFAULTS["enable_condenser"] is True

    assert "enable_condenser" in COMMIT0_INFER_DEFAULTS
    assert "condenser_max_size" in COMMIT0_INFER_DEFAULTS
    assert "condenser_keep_first" in COMMIT0_INFER_DEFAULTS
    assert COMMIT0_INFER_DEFAULTS["enable_condenser"] is True


def test_eval_metadata_accepts_condenser_params():
    """Test that EvalMetadata accepts condenser parameters."""
    llm = LLM(model="test-model", api_key="test-key")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test-dataset",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        critic=PassCritic(),
        enable_condenser=True,
        condenser_max_size=100,
        condenser_keep_first=5,
    )
    assert metadata.enable_condenser is True
    assert metadata.condenser_max_size == 100
    assert metadata.condenser_keep_first == 5


def test_eval_metadata_condenser_defaults():
    """Test that EvalMetadata uses correct defaults for condenser params."""
    llm = LLM(model="test-model", api_key="test-key")
    metadata = EvalMetadata(
        llm=llm,
        dataset="test-dataset",
        max_iterations=10,
        eval_output_dir="/tmp/test",
        critic=PassCritic(),
    )
    # Should use default values defined in EvalMetadata
    assert metadata.enable_condenser is True
    assert metadata.condenser_max_size == 240
    assert metadata.condenser_keep_first == 2


def test_args_parser_has_condenser_args():
    """Test that argument parser includes condenser arguments."""
    parser = get_parser(add_llm_config=False)
    # Parse empty args to get defaults
    args = parser.parse_args([])
    assert hasattr(args, "enable_condenser")
    assert hasattr(args, "disable_condenser")
    assert hasattr(args, "condenser_max_size")
    assert hasattr(args, "condenser_keep_first")


def test_condenser_enable_disable_flags():
    """Test that enable/disable condenser flags work correctly."""
    parser = get_parser(add_llm_config=False)

    # Test enable flag
    args = parser.parse_args(["--enable-condenser"])
    assert args.enable_condenser is True

    # Test disable flag
    args = parser.parse_args(["--disable-condenser"])
    assert args.disable_condenser is True

    # Test both flags (disable should take precedence in implementation)
    args = parser.parse_args(["--enable-condenser", "--disable-condenser"])
    assert args.enable_condenser is True
    assert args.disable_condenser is True


def test_condenser_size_args():
    """Test that condenser size arguments can be set."""
    parser = get_parser(add_llm_config=False)
    args = parser.parse_args(
        ["--condenser-max-size", "120", "--condenser-keep-first", "10"]
    )
    assert args.condenser_max_size == 120
    assert args.condenser_keep_first == 10
