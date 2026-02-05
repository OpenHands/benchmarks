"""Tests for modal argument parsing in eval_infer CLIs."""

import argparse

import pytest


class TestSwebenchEvalModalArgument:
    """Tests for swebench eval_infer modal argument parsing."""

    def test_modal_default_is_true(self):
        """Test that modal defaults to True when no flag is provided."""

        # Create a minimal parser to test the argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--run-id", required=True)
        parser.add_argument(
            "--no-modal",
            dest="modal",
            action="store_false",
        )
        parser.set_defaults(modal=True)

        args = parser.parse_args(["test.jsonl", "--run-id", "test"])
        assert args.modal is True

    def test_no_modal_flag_sets_modal_to_false(self):
        """Test that --no-modal flag sets modal to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--run-id", required=True)
        parser.add_argument(
            "--no-modal",
            dest="modal",
            action="store_false",
        )
        parser.set_defaults(modal=True)

        args = parser.parse_args(["test.jsonl", "--run-id", "test", "--no-modal"])
        assert args.modal is False

    def test_modal_flag_not_accepted(self):
        """Test that --modal flag is not accepted (removed from argparse)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--run-id", required=True)
        parser.add_argument(
            "--no-modal",
            dest="modal",
            action="store_false",
        )
        parser.set_defaults(modal=True)

        with pytest.raises(SystemExit):
            parser.parse_args(["test.jsonl", "--run-id", "test", "--modal"])


class TestSwebenchMultimodalEvalModalArgument:
    """Tests for swebenchmultimodal eval_infer modal argument parsing."""

    def test_modal_default_is_true(self):
        """Test that modal defaults to True when no flag is provided."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument(
            "--no-modal",
            dest="modal",
            action="store_false",
        )
        parser.set_defaults(modal=True)

        args = parser.parse_args(["test.jsonl"])
        assert args.modal is True

    def test_no_modal_flag_sets_modal_to_false(self):
        """Test that --no-modal flag sets modal to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument(
            "--no-modal",
            dest="modal",
            action="store_false",
        )
        parser.set_defaults(modal=True)

        args = parser.parse_args(["test.jsonl", "--no-modal"])
        assert args.modal is False


class TestSwebenchConfigNoModal:
    """Tests to verify modal is not in EVAL_DEFAULTS config."""

    def test_modal_not_in_swebench_eval_defaults(self):
        """Test that modal is not in EVAL_DEFAULTS for swebench."""
        from benchmarks.swebench.config import EVAL_DEFAULTS

        assert "modal" not in EVAL_DEFAULTS

    def test_swebench_eval_defaults_has_expected_keys(self):
        """Test that EVAL_DEFAULTS has the expected keys without modal."""
        from benchmarks.swebench.config import EVAL_DEFAULTS

        expected_keys = {"dataset", "split", "workers", "timeout"}
        assert set(EVAL_DEFAULTS.keys()) == expected_keys


class TestBackwardCompatibility:
    """Tests to verify backward compatibility of the changes."""

    def test_run_swebench_evaluation_default_modal_true(self):
        """Test that run_swebench_evaluation defaults modal to True."""
        import inspect

        from benchmarks.swebench.eval_infer import run_swebench_evaluation

        sig = inspect.signature(run_swebench_evaluation)
        modal_param = sig.parameters.get("modal")
        assert modal_param is not None
        assert modal_param.default is True

    def test_run_swebench_multimodal_evaluation_default_modal_true(self):
        """Test that run_swebench_multimodal_evaluation defaults modal to True."""
        import inspect

        from benchmarks.swebenchmultimodal.eval_infer import (
            run_swebench_multimodal_evaluation,
        )

        sig = inspect.signature(run_swebench_multimodal_evaluation)
        modal_param = sig.parameters.get("modal")
        assert modal_param is not None
        assert modal_param.default is True
