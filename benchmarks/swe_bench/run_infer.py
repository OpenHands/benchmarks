from __future__ import annotations

import os

from pydantic import SecretStr

from benchmarks.utils.agent_server import run_remote_evaluation
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.run_evaluation import (
    construct_eval_output_dir,
    make_metadata,
)
from openhands.sdk import LLM, get_logger


logger = get_logger(__name__)


def main():
    default_prompt_path = os.path.join(
        os.path.dirname(__file__), "prompts", "default.j2"
    )
    parser = get_parser()
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=default_prompt_path,
        help="Path to prompt template file",
    )
    args = parser.parse_args()

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
        service_id="litellm_proxy",
    )

    dataset_description = DATASET.replace("/", "__") + "-" + SPLIT.replace("/", "__")

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
        details={},
        dataset=DATASET,
        data_split=SPLIT,
        prompt_path=PROMPT_PATH,
        eval_n_limit=EVAL_N_LIMIT,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
    )

    # Always use remote evaluation
    run_remote_evaluation(llm, metadata, args.eval_num_workers)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
