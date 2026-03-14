"""Tests for add_prompt_path_argument utility."""

import argparse
from pathlib import Path

import pytest

from benchmarks.utils.args_parser import add_prompt_path_argument


@pytest.fixture()
def prompt_tree(tmp_path: Path) -> Path:
    """Create a minimal benchmark directory with a prompts/ sub-directory."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.j2").write_text("{{ task }}")
    (prompts_dir / "custom.j2").write_text("{{ task }} (custom)")
    # Fake caller module living next to the prompts/ dir.
    caller = tmp_path / "run_infer.py"
    caller.write_text("")
    return caller


class TestAddPromptPathArgument:
    """Test that --prompt-path resolves correctly in all scenarios."""

    def test_default_value_is_absolute(self, prompt_tree: Path) -> None:
        parser = argparse.ArgumentParser()
        add_prompt_path_argument(parser, str(prompt_tree))
        args = parser.parse_args([])
        assert Path(args.prompt_path).is_absolute()
        assert args.prompt_path.endswith("default.j2")

    def test_bare_filename_resolves_to_absolute(self, prompt_tree: Path) -> None:
        parser = argparse.ArgumentParser()
        add_prompt_path_argument(parser, str(prompt_tree))
        args = parser.parse_args(["--prompt-path", "custom.j2"])
        assert Path(args.prompt_path).is_absolute()
        assert args.prompt_path.endswith("custom.j2")

    def test_absolute_path_accepted(self, prompt_tree: Path) -> None:
        abs_path = str(prompt_tree.parent / "prompts" / "custom.j2")
        parser = argparse.ArgumentParser()
        add_prompt_path_argument(parser, str(prompt_tree))
        args = parser.parse_args(["--prompt-path", abs_path])
        assert args.prompt_path == abs_path

    def test_works_from_different_cwd(
        self, prompt_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original bug: ValueError when CWD != project root."""
        # Change CWD to an unrelated directory.
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        parser = argparse.ArgumentParser()
        add_prompt_path_argument(parser, str(prompt_tree))
        args = parser.parse_args([])
        assert Path(args.prompt_path).is_file()

    def test_invalid_template_gives_clear_error(self, prompt_tree: Path) -> None:
        parser = argparse.ArgumentParser()
        add_prompt_path_argument(parser, str(prompt_tree))
        with pytest.raises(SystemExit):
            parser.parse_args(["--prompt-path", "nonexistent.j2"])

    def test_missing_default_template_raises(self, tmp_path: Path) -> None:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "other.j2").write_text("no default")
        caller = tmp_path / "run_infer.py"
        caller.write_text("")

        parser = argparse.ArgumentParser()
        with pytest.raises(AssertionError, match="default.j2"):
            add_prompt_path_argument(parser, str(caller))

    def test_relative_path_accepted(
        self, prompt_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backwards compatibility: relative paths that exist are resolved."""
        monkeypatch.chdir(prompt_tree.parent)
        parser = argparse.ArgumentParser()
        add_prompt_path_argument(parser, str(prompt_tree))
        args = parser.parse_args(["--prompt-path", "prompts/custom.j2"])
        assert Path(args.prompt_path).is_absolute()
        assert args.prompt_path.endswith("custom.j2")
