import argparse
import json
import os

from benchmarks.multiswebench.constants import (
    DEFAULT_CLEAR_ENV,
    DEFAULT_EVAL_MODE,
    DEFAULT_FORCE_BUILD,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MAX_WORKERS_BUILD_IMAGE,
    DEFAULT_MAX_WORKERS_RUN_INSTANCE,
    DEFAULT_NEED_CLONE,
    DEFAULT_STOP_ON_ERROR,
    FIX_PATCH_RUN_CMD,
)
from benchmarks.multiswebench.scripts.eval.convert import convert_to_eval_format


def update_multi_swe_config(output_jsonl_path, config_path, dataset):
    path_to_parent = os.path.dirname(os.path.abspath(output_jsonl_path))
    converted_path = os.path.join(path_to_parent, "output_converted.jsonl")

    # Run the conversion function
    convert_to_eval_format(output_jsonl_path, converted_path)

    # Create required directories
    os.makedirs(os.path.join(path_to_parent, "eval_files", "dataset"), exist_ok=True)
    os.makedirs(os.path.join(path_to_parent, "eval_files", "workdir"), exist_ok=True)
    os.makedirs(os.path.join(path_to_parent, "eval_files", "repos"), exist_ok=True)
    os.makedirs(os.path.join(path_to_parent, "eval_files", "logs"), exist_ok=True)

    # Prepare config dict
    config = {
        "mode": DEFAULT_EVAL_MODE,
        "workdir": os.path.join(path_to_parent, "eval_files", "workdir"),
        "patch_files": [converted_path],
        "dataset_files": [dataset],
        "force_build": DEFAULT_FORCE_BUILD,
        "output_dir": os.path.join(path_to_parent, "eval_files", "dataset"),
        "specifics": [],
        "skips": [],
        "repo_dir": os.path.join(path_to_parent, "eval_files", "repos"),
        "need_clone": DEFAULT_NEED_CLONE,
        "global_env": [],
        "clear_env": DEFAULT_CLEAR_ENV,
        "stop_on_error": DEFAULT_STOP_ON_ERROR,
        "max_workers": DEFAULT_MAX_WORKERS,
        "max_workers_build_image": DEFAULT_MAX_WORKERS_BUILD_IMAGE,
        "max_workers_run_instance": DEFAULT_MAX_WORKERS_RUN_INSTANCE,
        "log_dir": os.path.join(path_to_parent, "eval_files", "logs"),
        "log_level": DEFAULT_LOG_LEVEL,
        "fix_patch_run_cmd": FIX_PATCH_RUN_CMD,
    }

    # Save to multibench.config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to create config")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    args = parser.parse_args()

    update_multi_swe_config(args.input, args.output, args.dataset)
