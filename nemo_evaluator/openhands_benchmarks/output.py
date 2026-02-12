import json
import pathlib

from nemo_evaluator.api.api_dataclasses import EvaluationResult


def parse_output(output_dir: str) -> EvaluationResult:
    output_path = pathlib.Path(output_dir)

    # Find any .report.json file (all benchmarks use this naming convention)
    report_files = sorted(output_path.rglob("*.report.json"))

    if not report_files:
        raise FileNotFoundError(
            f"No .report.json file found under {output_dir}. "
            "Make sure the evaluation completed successfully."
        )

    if len(report_files) > 1:
        raise ValueError(
            f"Multiple .report.json files found: {report_files}. "
            "`output_dir` must contain a single evaluation run."
        )

    report = json.loads(report_files[0].read_text(encoding="utf-8"))

    # Get benchmark name from report
    task_name = report["benchmark"]

    # All benchmarks have these common fields in their report
    resolved = report.get("resolved_instances", 0)
    submitted = report.get("submitted_instances", 0)

    # Calculate accuracy (handle division by zero)
    accuracy = resolved / submitted if submitted > 0 else 0.0

    metrics = {
        "accuracy": {
            "scores": {
                "accuracy": {
                    "value": accuracy,
                    "stats": {
                        "resolved": resolved,
                        "total": submitted,
                    },
                }
            }
        }
    }

    tasks = {task_name: {"metrics": metrics}}
    groups = {task_name: {"metrics": metrics}}

    return EvaluationResult(tasks=tasks, groups=groups)
