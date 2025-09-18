from __future__ import annotations

import json
import os

from pydantic import SecretStr

from benchmarks.utils.args_parser import parse_args
from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.run_evaluation import (
    construct_eval_output_dir,
    get_instruction,
    make_metadata,
    process_instance_simplified,
)
from benchmarks.utils.runtime import Runtime
from openhands.sdk import (
    LLM,
    get_logger,
)


logger = get_logger(__name__)


def main():
    default_prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", "default.j2"
    )
    args = parse_args(default_prompt_path)

    # SWT-specific defaults
    if not hasattr(args, "dataset") or not args.dataset:
        args.dataset = "princeton-nlp/SWE-bench"
    if not hasattr(args, "split") or not args.split:
        args.split = "test"

    DATASET = args.dataset
    SPLIT = args.split
    MODEL = args.llm_config
    EVAL_OUTPUT_DIR = args.eval_output_dir
    MAX_ITERATIONS = args.max_iterations
    EVAL_N_LIMIT = args.eval_n_limit
    EVAL_NOTE = args.eval_note
    PROMPT_PATH = args.prompt_path

    # Create LLM instance
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is not set")
    llm = LLM(
        model=MODEL,
        api_key=SecretStr(api_key),
        base_url="https://llm-proxy.eval.all-hands.dev",
        temperature=0,
    )

    dataset_description = (
        DATASET.replace("/", "__") + "-" + SPLIT.replace("/", "__") + "-swt"
    )

    # Construct proper structured output directory path
    structured_output_dir = construct_eval_output_dir(
        base_dir=EVAL_OUTPUT_DIR,
        dataset_name=dataset_description,
        model=llm.model,
        max_iterations=MAX_ITERATIONS,
        eval_note=EVAL_NOTE,
    )

    metadata = make_metadata(
        llm,
        dataset_description,
        MAX_ITERATIONS,
        structured_output_dir,
        details={"mode": "swt"},
        dataset=DATASET,
        data_split=SPLIT,
        prompt_path=PROMPT_PATH,
        eval_n_limit=EVAL_N_LIMIT,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
    )

    # Global variables for runtime methods
    global instances, output_file, results
    instances = None
    output_file = None
    results = []

    def initialize_runtime():
        """Initialize the runtime and retrieve instances to process."""
        global instances, output_file
        output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")

        # Prepare output file
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if dirname is not empty
            os.makedirs(output_dir, exist_ok=True)

        # Create empty output file
        with open(output_file, "w") as f:
            pass

        # Retrieve instances to process
        instances = get_dataset(
            metadata.dataset, metadata.data_split, output_file, metadata.eval_n_limit
        )
        print(f"### OUTPUT FILE: {output_file} ###")
        return instances

    def process_instance(instance):
        """Process a single instance."""
        global results, output_file
        logger.info(f"Processing instance {instance.instance_id}")

        # Get instruction
        workspace_path = os.path.join("/workspace", instance.repo.split("/")[-1])
        instruction = get_instruction(
            instance, metadata, workspace_path, metadata.prompt_path
        )
        result = process_instance_simplified(instance, instruction, metadata)

        # Save result using the complete format
        result_dict = result.model_dump(mode="json")
        if result.error:
            result_dict["error"] = result.error
        results.append(result_dict)

        logger.info(f"Writing result for {instance.instance_id} to {output_file}")
        logger.info(f"Result dict keys: {list(result_dict.keys())}")
        git_patch_len = len(result_dict.get("test_result", {}).get("git_patch", ""))
        logger.info(f"Git patch length: {git_patch_len}")

        # Write to output file
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "a") as f:
            json_line = json.dumps(result_dict) + "\n"
            f.write(json_line)
            f.flush()  # Ensure it's written immediately
            logger.info(
                f"Successfully wrote {len(json_line)} characters to output file"
            )

    def complete_runtime():
        """Complete the runtime - any cleanup if needed."""
        logger.info("SWT Runtime completed successfully!")

    # Create and run the Runtime
    runtime = Runtime(
        metadata=metadata,
        initialize_runtime=initialize_runtime,
        process_instance=process_instance,
        complete_runtime=complete_runtime,
    )

    runtime.run()

    logger.info("SWT Evaluation completed!")


if __name__ == "__main__":
    main()
