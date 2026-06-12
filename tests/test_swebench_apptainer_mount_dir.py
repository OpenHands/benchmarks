from pathlib import Path

from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.models import EvalInstance


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
