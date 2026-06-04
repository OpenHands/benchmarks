import pytest

from benchmarks.hybridgym_depsearch.run_infer import DepSearchEvaluation
from benchmarks.hybridgym_funcgen.run_infer import FuncGenEvaluation
from benchmarks.hybridgym_funclocalize.run_infer import FuncLocalizeEvaluation
from benchmarks.hybridgym_issuelocalize.run_infer import IssueLocalizeEvaluation
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.tool_presets import get_tools_for_preset
from openhands.tools.preset.gemini import get_gemini_tools
from openhands.tools.preset.gpt5 import get_gpt5_tools


def test_shared_tool_helper_supports_gemini_file_tools():
    expected = [tool.name for tool in get_gemini_tools(enable_browser=False)]
    actual = [
        tool.name for tool in get_tools_for_preset("gemini", enable_browser=False)
    ]

    assert actual == expected
    assert {"read_file", "write_file", "edit", "list_directory"}.issubset(actual)
    assert "file_editor" not in actual


def test_shared_tool_helper_supports_gpt5():
    expected = [tool.name for tool in get_gpt5_tools(enable_browser=False)]

    assert [
        tool.name for tool in get_tools_for_preset("gpt5", enable_browser=False)
    ] == expected


@pytest.mark.parametrize(
    "evaluation_cls",
    [
        DepSearchEvaluation,
        FuncGenEvaluation,
        FuncLocalizeEvaluation,
        IssueLocalizeEvaluation,
    ],
)
def test_hybridgym_tool_helpers_support_gpt5(evaluation_cls):
    expected = [tool.name for tool in get_gpt5_tools(enable_browser=False)]

    assert [
        tool.name for tool in evaluation_cls._get_tools(None, preset="gpt5")
    ] == expected


def test_common_parser_accepts_gpt5_tool_preset():
    parser = get_parser(add_llm_config=False)

    args = parser.parse_args(["--tool-preset", "gpt5"])

    assert args.tool_preset == "gpt5"
