"""
Argument parsing utilities for benchmarks.

This module defines common arguments used across all benchmarks.
Benchmark-specific defaults should be set via parser.set_defaults()
to match the evaluation repository configuration.
"""

import argparse
from pathlib import Path

from benchmarks.utils.critics import add_critic_args


def get_parser(add_llm_config: bool = True) -> argparse.ArgumentParser:
    """Create and return argument parser without defaults.

    Each benchmark must call parser.set_defaults() before parse_args()
    to set values matching the evaluation repository (OpenHands/evaluation).

    Args:
        add_llm_config: Whether to add the llm_config_path positional argument.

    Returns:
        ArgumentParser instance with common benchmark arguments (no defaults).
    """
    parser = argparse.ArgumentParser(description="Run Evaluation inference")
    if add_llm_config:
        parser.add_argument(
            "llm_config_path",
            type=str,
            help="Path to JSON LLM configuration",
        )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
    )
    parser.add_argument("--split", type=str, help="Dataset split")
    parser.add_argument(
        "--workspace",
        type=str,
        default="remote",
        choices=["docker", "remote"],
        help="Type of workspace to use (default: remote)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Maximum iterations (default: 500)",
    )
    parser.add_argument("--num-workers", type=int, help="Number of inference workers")
    parser.add_argument("--note", type=str, help="Optional evaluation note")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_outputs",
        help="Evaluation output directory",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit number of instances to evaluate (0 = no limit)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts for iterative mode (default: 3, min: 1)",
    )

    # Add critic arguments (no default)
    add_critic_args(parser)

    parser.add_argument(
        "--select",
        type=str,
        help="Path to text file containing instance IDs to select (one per line)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for instances that throw exceptions (default: 3)",
    )
    parser.add_argument(
        "--tool-preset",
        type=str,
        default="default",
        choices=["default", "gemini", "planning"],
        help=(
            "Tool preset for file editing. 'default' uses FileEditorTool, "
            "'gemini' uses read_file/write_file/edit/list_directory (default: default)"
        ),
    )
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=False,
        help="Enable sub-agent delegation tools for the agent",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="default",
        choices=["default", "acp-claude", "acp-codex"],
        help="Agent type: 'default' for standard Agent, 'acp-claude' for ACPAgent (Claude Code), 'acp-codex' for ACPAgent (Codex)",
    )
    return parser


def add_prompt_path_argument(parser: argparse.ArgumentParser, caller_file: str) -> None:
    """Add --prompt-path argument with choices from the benchmark's prompts/ dir.

    Resolves prompt templates relative to the caller's directory rather than
    CWD, so the argument works regardless of where the process is launched.

    Users can pass a bare filename (e.g. ``default.j2``), which is resolved
    against the benchmark's ``prompts/`` directory, or a full path to any
    ``.j2`` file for backwards compatibility.  The parsed value is always an
    absolute path so downstream code can rely on it directly.

    Args:
        parser: The argument parser to add the argument to.
        caller_file: Pass ``__file__`` from the calling module so we can
            locate its sibling ``prompts/`` directory.
    """
    prompt_dir = (Path(caller_file).parent / "prompts").resolve()
    templates = sorted(p.name for p in prompt_dir.glob("*.j2"))
    assert (prompt_dir / "default.j2").exists(), (
        f"Default prompt {prompt_dir / 'default.j2'} not found"
    )

    def _resolve_prompt(value: str) -> str:
        """Resolve a filename or path to an absolute prompt template path."""
        # Accept bare filenames (e.g. "default.j2") and resolve them.
        candidate = prompt_dir / Path(value).name
        if candidate.is_file():
            return str(candidate)
        # Also accept absolute/relative paths for backwards compatibility.
        p = Path(value)
        if p.is_file():
            return str(p.resolve())
        raise argparse.ArgumentTypeError(
            f"Prompt template not found: {value!r}. Available: {', '.join(templates)}"
        )

    parser.add_argument(
        "--prompt-path",
        type=_resolve_prompt,
        default=str(prompt_dir / "default.j2"),
        metavar="{" + ",".join(templates) + "}",
        help="Prompt template filename (default: default.j2)",
    )
