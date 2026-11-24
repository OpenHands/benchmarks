from contextlib import chdir
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID
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
    if report_path.exists() or not modal:
        return report_path

    prediction_rows = get_predictions_from_file(str(predictions_file), dataset, split)
    predictions = {row[KEY_INSTANCE_ID]: row for row in prediction_rows}
    full_dataset = load_swebench_dataset(dataset, split)
    with chdir(predictions_file.parent):
        generated_path = make_run_report(predictions, full_dataset, run_id)

    if generated_path.is_absolute():
        return generated_path
    return predictions_file.parent / generated_path
