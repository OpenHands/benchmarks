"""Tests for swtbench image_utils.py to verify correct dataset loading parameters."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch


def test_compute_required_images_uses_correct_is_swt_parameter(tmp_path, monkeypatch):
    """
    Test that compute_required_images uses is_swt=False when loading SWE-bench datasets.

    This is a regression test for issue #588 where using is_swt=True with SWE-bench
    datasets caused incorrect data transformation, resulting in only 212 instances
    being evaluated instead of the expected ~433.

    The is_swt parameter controls which transformation is applied:
    - is_swt=False → swe_to_swt_instance (for SWE-bench datasets)
    - is_swt=True → swt_to_swt_instance (for SWT-bench datasets with <patch> tags)

    Using the wrong transformation corrupts the data and causes evaluation failures.
    """
    from benchmarks.swtbench.image_utils import compute_required_images

    # Create a mock output.jsonl with some instance IDs
    output_jsonl = tmp_path / "output.jsonl"
    test_instance_ids = [
        "django__django-11333",
        "astropy__astropy-12345",
        "requests__requests-5555",
    ]

    with output_jsonl.open("w") as f:
        for instance_id in test_instance_ids:
            f.write(json.dumps({"instance_id": instance_id}) + "\n")

    # Mock the swt-bench repo directory
    swt_bench_dir = tmp_path / "swt-bench"
    swt_bench_dir.mkdir()
    (swt_bench_dir / "src").mkdir()
    (swt_bench_dir / "dataset").mkdir()

    # Create filter file (required by swt-bench code)
    filter_file = swt_bench_dir / "dataset" / "filter_cases_verified.txt"
    filter_file.write_text("")  # Empty filter for this test

    # Track what parameters were passed to load_swebench_dataset
    load_calls = []

    def mock_load_swebench_dataset(name, split, is_swt, filter_swt):
        """Mock that captures the parameters and returns mock data."""
        load_calls.append(
            {
                "name": name,
                "split": split,
                "is_swt": is_swt,
                "filter_swt": filter_swt,
            }
        )

        # Return mock dataset entries
        return [
            {"instance_id": iid, "base_commit": "abc123"} for iid in test_instance_ids
        ]

    def mock_make_exec_spec(entry):
        """Mock ExecSpec creation."""
        return SimpleNamespace(
            base_image_key=f"base-{entry['instance_id']}",
            env_image_key=f"env-{entry['instance_id']}",
        )

    # Patch the swt-bench functions
    with patch("benchmarks.swtbench.image_utils.ensure_swt_bench_repo") as mock_ensure:
        mock_ensure.return_value = swt_bench_dir

        # We need to patch sys.modules to inject our mocks
        mock_dataset_module = SimpleNamespace(
            load_swebench_dataset=mock_load_swebench_dataset
        )
        mock_exec_spec_module = SimpleNamespace(make_exec_spec=mock_make_exec_spec)

        monkeypatch.setitem(sys.modules, "src.dataset", mock_dataset_module)
        monkeypatch.setitem(sys.modules, "src.exec_spec", mock_exec_spec_module)

        # Call compute_required_images
        base_images, env_images = compute_required_images(
            output_jsonl,
            dataset="princeton-nlp/SWE-bench_Verified",
            split="test",
        )

    # Verify that load_swebench_dataset was called with the correct parameters
    assert len(load_calls) == 1, "load_swebench_dataset should be called once"

    call = load_calls[0]
    assert call["name"] == "princeton-nlp/SWE-bench_Verified"
    assert call["split"] == "test"
    assert call["filter_swt"] is True, (
        "filter_swt should be True to filter out problematic instances"
    )

    # This is the critical assertion - is_swt must be False for SWE-bench datasets
    assert call["is_swt"] is False, (
        "is_swt must be False when loading SWE-bench datasets. "
        "Using is_swt=True applies swt_to_swt_instance transformation "
        "which expects <patch> tags that don't exist in SWE-bench data, "
        "causing data corruption and evaluation failures (issue #588)."
    )

    # Verify that images were computed
    assert len(base_images) == 3
    assert len(env_images) == 3


def test_build_eval_env_images_uses_correct_is_swt_parameter():
    """
    Verify that build_eval_env_images.load_exec_specs uses is_swt=False.

    This test documents the correct behavior that should match compute_required_images.
    Both functions should use is_swt=False when loading SWE-bench datasets.
    """
    # This test verifies the code is correct by inspection
    # We check that load_exec_specs uses is_swt=False in its call to load_swebench_dataset
    import inspect

    from benchmarks.swtbench.build_eval_env_images import load_exec_specs

    source = inspect.getsource(load_exec_specs)

    # Verify the function contains the correct call
    assert "is_swt=False" in source or "is_swt = False" in source, (
        "load_exec_specs should use is_swt=False when loading SWE-bench datasets"
    )


def test_is_swt_parameter_consistency():
    """
    Ensure both image_utils and build_eval_env_images use consistent is_swt parameters.

    This test verifies that the fix for issue #588 maintains consistency between
    the two modules that load SWE-bench datasets.
    """
    import inspect

    from benchmarks.swtbench.build_eval_env_images import load_exec_specs
    from benchmarks.swtbench.image_utils import compute_required_images

    # Get source code for both functions
    compute_source = inspect.getsource(compute_required_images)
    load_source = inspect.getsource(load_exec_specs)

    # Both should use is_swt=False for SWE-bench datasets
    # (We check for "is_swt" followed by "False" allowing for variations in spacing)
    import re

    compute_has_false = bool(re.search(r"is_swt\s*[=:]\s*False", compute_source))
    load_has_false = bool(re.search(r"is_swt\s*[=:]\s*False", load_source))

    assert compute_has_false, (
        "compute_required_images should use is_swt=False for SWE-bench datasets"
    )
    assert load_has_false, (
        "load_exec_specs should use is_swt=False for SWE-bench datasets"
    )

    # Make sure neither uses is_swt=True in the critical load_swebench_dataset call
    compute_has_true = bool(
        re.search(
            r"load_swebench_dataset.*is_swt\s*=\s*True", compute_source, re.DOTALL
        )
    )
    load_has_true = bool(
        re.search(r"load_swebench_dataset.*is_swt\s*=\s*True", load_source, re.DOTALL)
    )

    assert not compute_has_true, (
        "compute_required_images should NOT use is_swt=True "
        "(this was the bug in issue #588)"
    )
    assert not load_has_true, (
        "load_exec_specs should NOT use is_swt=True "
        "for consistency with compute_required_images"
    )
