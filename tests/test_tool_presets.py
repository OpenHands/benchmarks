import pytest

import benchmarks.swebench.run_infer as swebench_run_infer
from benchmarks.hybridgym_depsearch.run_infer import DepSearchEvaluation
from benchmarks.hybridgym_funcgen.run_infer import FuncGenEvaluation
from benchmarks.hybridgym_funclocalize.run_infer import FuncLocalizeEvaluation
from benchmarks.hybridgym_issuelocalize.run_infer import IssueLocalizeEvaluation
from benchmarks.swebench.run_infer import (
    get_system_prompt_filename_for_preset as get_swebench_system_prompt,
    get_tools_for_preset as get_swebench_tools,
)
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


class _FakePromptTemplate:
    def __init__(self, exists: bool):
        self._exists = exists

    def joinpath(self, _filename: str):
        return self

    def is_file(self) -> bool:
        return self._exists


def test_swebench_uses_gpt5_system_prompt_when_available(monkeypatch):
    monkeypatch.setattr(
        swebench_run_infer.resources,
        "files",
        lambda _package: _FakePromptTemplate(True),
    )

    assert get_swebench_system_prompt("gpt5") == "system_prompt_gpt_5_4.j2"


def test_swebench_falls_back_when_gpt5_system_prompt_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        swebench_run_infer.resources,
        "files",
        lambda _package: _FakePromptTemplate(False),
    )

    assert get_swebench_system_prompt("gpt5") is None


@pytest.mark.parametrize("preset", ["default", "gemini", "planning"])
def test_swebench_non_gpt5_presets_do_not_override_system_prompt(preset):
    assert get_swebench_system_prompt(preset) is None
