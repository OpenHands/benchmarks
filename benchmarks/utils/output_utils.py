"""Utilities for processing and enriching output files."""

from __future__ import annotations

import json
from pathlib import Path


def add_resolve_rate_to_predictions(
    predictions_path: str | Path,
    report_path: str | Path,
) -> None:
    """
    Add resolution status from a report to each prediction in a JSONL file.

    For each prediction in the predictions file, if the instance_id is found in
    the report's resolved_ids or unresolved_ids, a "report" field is added with
    {"resolved": true/false}. Predictions not found in either list are left unchanged.

    Args:
        predictions_path: Path to the predictions JSONL file. Each line should be
            a JSON object with an "instance_id" field.
        report_path: Path to the report JSON file containing "resolved_ids" and
            "unresolved_ids" lists.
    """
    predictions_path = Path(predictions_path)
    report_path = Path(report_path)

    # Load the report
    with open(report_path, "r") as f:
        report = json.load(f)

    resolved_ids = set(report.get("resolved_ids", []))
    unresolved_ids = set(report.get("unresolved_ids", []))

    # Read all predictions
    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    # Update predictions with resolution status
    updated_predictions = []
    for prediction in predictions:
        instance_id = prediction.get("instance_id")
        if instance_id in resolved_ids:
            prediction["report"] = {"resolved": True}
        elif instance_id in unresolved_ids:
            prediction["report"] = {"resolved": False}
        # If not in either list, leave unchanged (no report field added)
        updated_predictions.append(prediction)

    # Write back to the same file
    with open(predictions_path, "w") as f:
        for prediction in updated_predictions:
            f.write(json.dumps(prediction) + "\n")
