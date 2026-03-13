"""Tests for shared workspace type configuration."""

from benchmarks.utils.args_parser import get_parser


def test_parser_accepts_apptainer_workspace() -> None:
    parser = get_parser(add_llm_config=False)
    args = parser.parse_args(["--workspace", "apptainer"])
    assert args.workspace == "apptainer"
