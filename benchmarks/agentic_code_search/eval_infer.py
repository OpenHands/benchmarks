import json
from argparse import ArgumentParser

from datasets import load_dataset


filtered_instances = [
    "pytest-dev__pytest-5227",
    "sympy__sympy-15345",
    "sympy__sympy-21614",
    "scikit-learn__scikit-learn-13439",
    "sympy__sympy-11400",
    "sympy__sympy-19487",
    "sympy__sympy-15308",
    "django__django-12915",
    "sympy__sympy-20590",
    "sympy__sympy-17022",
    "django__django-11099",
    "django__django-13220",
    "django__django-11964",
    "matplotlib__matplotlib-25332",
    "django__django-10914",
    "django__django-14915",
    "django__django-11049",
    "django__django-11564",
    "sympy__sympy-17655",
    "sympy__sympy-16106",
    "sympy__sympy-12171",
    "django__django-15400",
    "django__django-14411",
    "sympy__sympy-21055",
    "django__django-15213",
    "django__django-15902",
]


def read_jsonl(filename):
    data = {}
    stats = {}
    with open(filename, "r") as f:
        for line in f:
            instance = json.loads(line.strip())
            instance_id = instance["instance_id"]
            if instance_id in filtered_instances and args.variant == "Lite":
                continue
            test_result = instance.get("test_result", {})
            reward = test_result.get("reward", {})
            prediction = reward.get("prediction", {})
            if len(prediction) == 0:
                prediction = {"files": [], "modules": [], "entities": []}
            data[instance_id] = prediction
            stats[instance_id] = {
                "num_steps": test_result.get("num_steps", 0),
                "num_tool_calls": test_result.get("num_tool_calls", 0),
                "wall_time_seconds": test_result.get("wall_time_seconds", 0),
            }
    return data, stats


def read_hf_dataset(dataset_name, split="test"):
    dataset = load_dataset(dataset_name, split=split)
    data = {}
    for instance in dataset:
        instance_id: str = instance["instance_id"]  # type: ignore
        if instance_id in filtered_instances and args.variant == "Lite":
            continue
        gt_files = []
        gt_modules = []
        gt_entities = []
        file_changes = instance["file_changes"]  # type: ignore
        if file_changes is None:
            file_changes = []
        for change in file_changes:
            if "file" in change:
                gt_files.append(change["file"])
            if "changes" in change:
                edited_modules = change["changes"].get("edited_modules", [])
                edited_modules = [] if edited_modules is None else edited_modules
                for module in edited_modules:
                    gt_modules.append(module)

                edited_entities = change["changes"].get("edited_entities", [])
                edited_entities = [] if edited_entities is None else edited_entities
                for entity in edited_entities:
                    gt_entities.append(entity)
        gt_files = list(set(gt_files))
        gt_modules = list(set(gt_modules))
        gt_entities = list(set(gt_entities))
        data[instance_id] = {
            "files": gt_files,
            "modules": gt_modules,
            "entities": gt_entities,
        }
    return data


def compute_precision(predicted_set, true_set):
    """Compute precision: TP / (TP + FP) = TP / |predicted|"""
    if len(predicted_set) == 0:
        return 0.0
    tp = len(predicted_set & true_set)
    return tp / len(predicted_set)


def compute_recall(predicted_set, true_set):
    """Compute recall: TP / (TP + FN) = TP / |true|"""
    if len(true_set) == 0:
        return 0.0
    tp = len(predicted_set & true_set)
    return tp / len(true_set)


def f1_reward_function(predicted_set, true_set):
    precision = compute_precision(predicted_set, true_set)
    recall = compute_recall(predicted_set, true_set)
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def compute_metrics(predictions, ground_truth, stats=None):
    all_instance_ids = set(predictions.keys()).union(set(ground_truth.keys()))
    # total_instances = len(all_instance_ids)

    # hard-code the total instances - locagent faces errors on 2 of SWE-Bench Pro instances due to which HF data has 264 instances only.
    # NOTE: Instances are 274 and not 300 for SWE-Bench Lite because we primarily compare against LocAgent baseline and they discard these instances from evals since their patch processing code produces empty ground truth outputs for these instances.
    TOTAL_INSTANCE_MAP = {"Lite": 274, "Pro": 266, "Verified": 500}
    total_instances = TOTAL_INSTANCE_MAP[args.variant]
    # print(len(ground_truth), len(predictions))
    print(f"Total instances to evaluate: {total_instances}")

    # Initialize accumulators for all metrics at all levels
    file_precision = 0
    file_recall = 0
    file_f1 = 0

    module_precision = 0
    module_recall = 0
    module_f1 = 0

    entity_precision = 0
    entity_recall = 0
    entity_f1 = 0

    num_steps = 0
    num_tool_calls = 0
    total_time = 0

    for instance_id in all_instance_ids:
        if instance_id not in predictions or instance_id not in ground_truth:
            continue

        pred = predictions[instance_id]
        gt = ground_truth[instance_id]

        # File-level metrics
        pred_files = set(pred.get("files", []))
        gt_files = set(gt.get("files", []))
        file_precision += compute_precision(pred_files, gt_files)
        file_recall += compute_recall(pred_files, gt_files)
        file_f1 += f1_reward_function(pred_files, gt_files)

        # Module-level metrics
        pred_modules = set(pred.get("modules", []))
        gt_modules = set(gt.get("modules", []))
        module_precision += compute_precision(pred_modules, gt_modules)
        module_recall += compute_recall(pred_modules, gt_modules)
        module_f1 += f1_reward_function(pred_modules, gt_modules)

        # Entity-level metrics
        pred_entities = set(pred.get("entities", []))
        gt_entities = set(gt.get("entities", []))
        entity_precision += compute_precision(pred_entities, gt_entities)
        entity_recall += compute_recall(pred_entities, gt_entities)
        entity_f1 += f1_reward_function(pred_entities, gt_entities)

        # Step/time stats
        if stats and instance_id in stats:
            num_steps += stats[instance_id].get("num_steps", 0)
            num_tool_calls += stats[instance_id].get("num_tool_calls", 0)
            total_time += stats[instance_id].get("wall_time_seconds", 0)

    # Compute averages
    avg_file_precision = file_precision / total_instances
    avg_file_recall = file_recall / total_instances
    avg_file_f1 = file_f1 / total_instances

    avg_module_precision = module_precision / total_instances
    avg_module_recall = module_recall / total_instances
    avg_module_f1 = module_f1 / total_instances

    avg_entity_precision = entity_precision / total_instances
    avg_entity_recall = entity_recall / total_instances
    avg_entity_f1 = entity_f1 / total_instances

    # Print results in a formatted table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nFile-level Metrics:")
    print(f"  F1 Score:  {avg_file_f1:.4f}")
    print(f"  Precision: {avg_file_precision:.4f}")
    print(f"  Recall:    {avg_file_recall:.4f}")

    print("\nModule-level Metrics:")
    print(f"  F1 Score:  {avg_module_f1:.4f}")
    print(f"  Precision: {avg_module_precision:.4f}")
    print(f"  Recall:    {avg_module_recall:.4f}")

    print("\nFunction-level Metrics:")
    print(f"  F1 Score:  {avg_entity_f1:.4f}")
    print(f"  Precision: {avg_entity_precision:.4f}")
    print(f"  Recall:    {avg_entity_recall:.4f}")

    if stats:
        print("\nEfficiency Metrics:")
        print(f"  Avg # of steps:      {num_steps / total_instances:.4f}")
        print(f"  Avg # of tool calls: {num_tool_calls / total_instances:.4f}")
        print(f"  Avg wall time (s):   {total_time / total_instances:.4f}")

    print("\n" + "=" * 60)

    # Return results as a dictionary for programmatic access
    return {
        "file": {
            "precision": avg_file_precision,
            "recall": avg_file_recall,
            "f1": avg_file_f1,
        },
        "module": {
            "precision": avg_module_precision,
            "recall": avg_module_recall,
            "f1": avg_module_f1,
        },
        "entity": {
            "precision": avg_entity_precision,
            "recall": avg_entity_recall,
            "f1": avg_entity_f1,
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path the JSONL file containing the agent outputs",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["Lite", "Verified", "Pro"],
        help="Which variant of SWE-Bench test dataset to evaluate on (Lite, Verified, or Pro)",
    )
    args = parser.parse_args()

    hf_dataset = f"adityasoni17/SWE-bench_{args.variant}-locagent"
    output_file = args.output_file
    predictions, stats = read_jsonl(output_file)
    ground_truth = read_hf_dataset(hf_dataset, split="test")
    results = compute_metrics(predictions, ground_truth, stats)
