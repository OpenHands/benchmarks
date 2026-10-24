import json
from contextlib import chdir
from pathlib import Path

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    LOG_REPORT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.reporting import make_run_report
from swebench.harness.utils import get_predictions_from_file, load_swebench_dataset

from benchmarks.utils.constants import MODEL_NAME_OR_PATH


def ensure_swebench_run_report(
    predictions_file: Path,
    dataset: str,
    split: str,
    run_id: str,
    modal: bool,
) -> Path:
    """Return the aggregate report, building it after a Modal run if needed."""
    predictions_file = predictions_file.resolve()
    report_path = predictions_file.parent / f"{MODEL_NAME_OR_PATH}.{run_id}.json"
    if not modal:
        return report_path

    prediction_rows = get_predictions_from_file(str(predictions_file), dataset, split)
    for row in prediction_rows:
        instance_id = row[KEY_INSTANCE_ID]
        instance_report = (
            predictions_file.parent
            / RUN_EVALUATION_LOG_DIR
            / run_id
            / row[KEY_MODEL].replace("/", "__")
            / instance_id
            / LOG_REPORT
        )
        if instance_report.exists():
            result = json.loads(instance_report.read_text())[instance_id]
            if result.get("error"):
                raise RuntimeError(
                    f"Grading failed for {instance_id}: {result['error']}"
                )
    if report_path.exists():
        return report_path

    predictions = {row[KEY_INSTANCE_ID]: row for row in prediction_rows}
    full_dataset = load_swebench_dataset(dataset, split)
    with chdir(predictions_file.parent):
        generated_path = make_run_report(predictions, full_dataset, run_id)

    if generated_path.is_absolute():
        return generated_path
    return predictions_file.parent / generated_path
