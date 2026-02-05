"""Tests for modal argument parsing in eval_infer CLIs."""

import argparse

from benchmarks.swebench.config import EVAL_DEFAULTS


class TestSwebenchEvalModalArgument:
    """Tests for swebench eval_infer modal argument parsing."""

    def test_modal_default_from_eval_defaults(self):
        """Test that modal defaults to EVAL_DEFAULTS['modal'] when no flag is provided."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--modal", dest="modal", action="store_true")
        parser.add_argument("--no-modal", dest="modal", action="store_false")
        parser.set_defaults(**EVAL_DEFAULTS)

        args = parser.parse_args(["test.jsonl", "--run-id", "test"])
        assert args.modal == EVAL_DEFAULTS["modal"]

    def test_modal_flag_sets_modal_to_true(self):
        """Test that --modal flag sets modal to True."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--modal", dest="modal", action="store_true")
        parser.add_argument("--no-modal", dest="modal", action="store_false")
        parser.set_defaults(**EVAL_DEFAULTS)

        args = parser.parse_args(["test.jsonl", "--run-id", "test", "--modal"])
        assert args.modal is True

    def test_no_modal_flag_sets_modal_to_false(self):
        """Test that --no-modal flag sets modal to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--modal", dest="modal", action="store_true")
        parser.add_argument("--no-modal", dest="modal", action="store_false")
        parser.set_defaults(**EVAL_DEFAULTS)

        args = parser.parse_args(["test.jsonl", "--run-id", "test", "--no-modal"])
        assert args.modal is False


class TestSwebenchMultimodalEvalModalArgument:
    """Tests for swebenchmultimodal eval_infer modal argument parsing."""

    def test_modal_default_is_true(self):
        """Test that modal defaults to True when no flag is provided."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--no-modal", dest="modal", action="store_false")
        parser.set_defaults(modal=True)

        args = parser.parse_args(["test.jsonl"])
        assert args.modal is True

    def test_no_modal_flag_sets_modal_to_false(self):
        """Test that --no-modal flag sets modal to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file")
        parser.add_argument("--no-modal", dest="modal", action="store_false")
        parser.set_defaults(modal=True)

        args = parser.parse_args(["test.jsonl", "--no-modal"])
        assert args.modal is False


class TestSwebenchConfig:
    """Tests to verify modal is in EVAL_DEFAULTS config."""

    def test_modal_in_swebench_eval_defaults(self):
        """Test that modal is in EVAL_DEFAULTS for swebench."""
        assert "modal" in EVAL_DEFAULTS
        assert EVAL_DEFAULTS["modal"] is True

    def test_swebench_eval_defaults_has_expected_keys(self):
        """Test that EVAL_DEFAULTS has the expected keys including modal."""
        expected_keys = {"dataset", "split", "workers", "timeout", "modal"}
        assert set(EVAL_DEFAULTS.keys()) == expected_keys


class TestBackwardCompatibility:
    """Tests to verify backward compatibility of the changes."""

    def test_run_swebench_evaluation_default_modal_from_eval_defaults(self):
        """Test that run_swebench_evaluation defaults modal to EVAL_DEFAULTS['modal']."""
        import inspect

        from benchmarks.swebench.eval_infer import run_swebench_evaluation

        sig = inspect.signature(run_swebench_evaluation)
        modal_param = sig.parameters.get("modal")
        assert modal_param is not None
        assert modal_param.default == EVAL_DEFAULTS["modal"]

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
