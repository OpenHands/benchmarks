import pytest

from benchmarks.hybridgym_depsearch.run_infer import DepSearchEvaluation
from benchmarks.hybridgym_funcgen.run_infer import FuncGenEvaluation
from benchmarks.hybridgym_funclocalize.run_infer import FuncLocalizeEvaluation
from benchmarks.hybridgym_issuelocalize.run_infer import IssueLocalizeEvaluation
from benchmarks.swebench.run_infer import get_tools_for_preset as get_swebench_tools
from benchmarks.swebenchmultilingual.run_infer import (
    get_tools_for_preset as get_swebenchmultilingual_tools,
)
from benchmarks.utils.args_parser import get_parser
from openhands.tools.preset.gpt5 import get_gpt5_tools


@pytest.mark.parametrize(
    "getter",
    [
        get_swebench_tools,
        get_swebenchmultilingual_tools,
    ],
)
def test_swebench_tool_helpers_support_gpt5(getter):
    expected = [tool.name for tool in get_gpt5_tools(enable_browser=False)]

    assert [tool.name for tool in getter("gpt5", enable_browser=False)] == expected


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
