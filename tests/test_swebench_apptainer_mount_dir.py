from pathlib import Path
from types import SimpleNamespace

from benchmarks.swebench import run_infer as swebench_run_infer
from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.models import EvalInstance, EvalMetadata
from openhands.sdk import LLM
from openhands.sdk.critic import PassCritic


def test_apptainer_mount_dir_uses_writable_env_root(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHANDS_APPTAINER_WORKSPACE_ROOT", str(tmp_path))

    evaluation = object.__new__(SWEBenchEvaluation)
    object.__setattr__(evaluation, "current_attempt", 2)

    mount_dir = Path(
        evaluation.get_apptainer_mount_dir(
            EvalInstance(id="django__django-12345", data={})
        )
    )

    assert mount_dir.parent == tmp_path
    assert mount_dir.name.startswith("django__django-12345-attempt2-")
    assert mount_dir.is_dir()


def test_apptainer_mount_dir_uses_default_current_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENHANDS_APPTAINER_WORKSPACE_ROOT", str(tmp_path / "workspaces")
    )

    evaluation = SWEBenchEvaluation(
        metadata=EvalMetadata(
            llm=LLM(model="test-model"),
            dataset="test",
            max_iterations=1,
            eval_output_dir=str(tmp_path / "output"),
            details={},
            critic=PassCritic(),
        )
    )

    mount_dir = Path(
        evaluation.get_apptainer_mount_dir(
            EvalInstance(id="django__django-12345", data={})
        )
    )

    assert evaluation.current_attempt == 1
    assert mount_dir.parent == tmp_path / "workspaces"
    assert mount_dir.name.startswith("django__django-12345-attempt1-")
    assert mount_dir.is_dir()


def test_apptainer_mount_dir_retries_uuid_collision(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENHANDS_APPTAINER_WORKSPACE_ROOT", str(tmp_path))
    uuids = iter(
        [
            SimpleNamespace(hex="abc12345deadbeef"),
            SimpleNamespace(hex="def67890deadbeef"),
        ]
    )
    monkeypatch.setattr(swebench_run_infer.uuid, "uuid4", lambda: next(uuids))

    evaluation = object.__new__(SWEBenchEvaluation)
    object.__setattr__(evaluation, "current_attempt", 2)
    existing_mount_dir = tmp_path / "django__django-12345-attempt2-abc12345"
    existing_mount_dir.mkdir()

    mount_dir = Path(
        evaluation.get_apptainer_mount_dir(
            EvalInstance(id="django__django-12345", data={})
        )
    )

    assert mount_dir == tmp_path / "django__django-12345-attempt2-def67890"
    assert mount_dir.is_dir()
