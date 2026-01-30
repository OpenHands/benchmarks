import argparse
import json
import os

from benchmarks.multiswebench.constants import DEFAULT_EVAL_HARNESS_CONFIG
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

    # Start with default config and add dynamic paths
    config = DEFAULT_EVAL_HARNESS_CONFIG.copy()
    config["workdir"] = os.path.join(path_to_parent, "eval_files", "workdir")
    config["patch_files"] = [converted_path]
    config["dataset_files"] = [dataset]
    config["output_dir"] = os.path.join(path_to_parent, "eval_files", "dataset")
    config["repo_dir"] = os.path.join(path_to_parent, "eval_files", "repos")
    config["log_dir"] = os.path.join(path_to_parent, "eval_files", "logs")

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
