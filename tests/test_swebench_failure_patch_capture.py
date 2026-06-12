from types import SimpleNamespace
from typing import cast

from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.critics import PassCritic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata
from openhands.sdk import LLM
from openhands.sdk.workspace import RemoteWorkspace


class DummyEvaluation(Evaluation):
    def prepare_instances(self):
        return []

    def prepare_workspace(
        self,
        instance,
        resource_factor=1,
        forward_env=None,
    ):
        raise NotImplementedError

    def evaluate_instance(self, instance, workspace):
        raise NotImplementedError


class FakeWorkspace:
    def __init__(self):
        self.commands = []

    def execute_command(self, command, *args, **kwargs):
        self.commands.append(command)
        if "git --no-pager diff --no-color --cached abc123" in command:
            return SimpleNamespace(
                exit_code=0,
                stdout="diff --git a/file.py b/file.py\n",
                stderr="",
            )
        if "git --no-pager diff --no-color abc123 HEAD" in command:
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def _metadata(tmp_path):
    return EvalMetadata(
        llm=LLM(model="test-model", api_key="test-key"),
        dataset="test-dataset",
        max_iterations=1,
        eval_output_dir=str(tmp_path),
        critic=PassCritic(),
    )


def test_error_output_preserves_failure_test_result(tmp_path):
    evaluation = DummyEvaluation(metadata=_metadata(tmp_path))
    instance = EvalInstance(id="instance-1", data={})

    output = evaluation._create_error_output(
        instance,
        RuntimeError("boom"),
        retry_count=0,
        test_result={"git_patch": "diff --git a/file.py b/file.py\n"},
    )

    assert output.error is not None
    assert output.test_result["git_patch"].startswith("diff --git")


def test_swebench_collect_failure_test_result_gets_git_patch(tmp_path):
    evaluation = SWEBenchEvaluation(metadata=_metadata(tmp_path))
    instance = EvalInstance(
        id="django__django-12345",
        data={
            "repo": "django/django",
            "base_commit": "abc123",
        },
    )
    workspace = FakeWorkspace()

    result = evaluation.collect_failure_test_result(
        instance,
        cast(RemoteWorkspace, workspace),
        RuntimeError("Remote conversation got stuck"),
    )

    assert result == {
        "git_patch": "diff --git a/file.py b/file.py\n",
        "git_patch_captured_on_error": True,
    }
    assert workspace.commands == [
        "cd /workspace/django/ ; git add -A",
        "cd /workspace/django/ ; git --no-pager diff --no-color --cached abc123",
    ]
