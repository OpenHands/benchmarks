import os

import openhands.agenthub
from benchmarks.utils.commit0_evaluation import (
    commit0_setup,
    process_instance,
)
from benchmarks.utils.shared import (
    make_metadata,
    prepare_dataset,
    run_evaluation,
)
from commit0.harness.constants import SPLIT
from datasets import load_dataset
from openhands.core.config import (
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.logger import openhands_logger as logger


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='wentingzhao/commit0_combined',
        help='dataset to evaluate on, only test split exists for this HF dataset',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='this is the HF dataset split',
    )
    parser.add_argument(
        '--repo-split',
        type=str,
        default='lite',
        help='all, lite, or each repo name',
    )
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenHands's repo
    dataset = load_dataset(args.dataset, split=args.split)

    commit0_datasets = commit0_setup(dataset.to_pandas(), args.repo_split)

    logger.info(f'Loaded dataset {args.dataset} with reposplit {args.repo_split}')

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_descrption = (
        args.dataset.replace('/', '__') + '-' + args.repo_split.replace('/', '__')
    )
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    instances = prepare_dataset(commit0_datasets, output_file, args.eval_n_limit)

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=120 * 60,  # 2 hour PER instance should be more than enough
    )