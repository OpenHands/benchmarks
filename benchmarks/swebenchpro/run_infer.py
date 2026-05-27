import json

from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.swebenchpro import constants
from benchmarks.swebenchpro.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swebenchpro.config import INFER_DEFAULTS
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import EvalInstance, EvalMetadata
from openhands.sdk import get_logger


logger = get_logger(__name__)


class SWEBenchProEvaluation(SWEBenchEvaluation):
    def get_official_docker_image(self, instance: EvalInstance) -> str:
        return get_official_docker_image(instance.data)

    def extract_custom_tag(self, official_docker_image: str) -> str:
        return extract_custom_tag(official_docker_image)

    def should_wrap_instance(self, instance: EvalInstance) -> bool:
        return False

    def get_source_repo_path(self, instance: EvalInstance) -> str:
        return constants.SOURCE_REPO_PATH


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    critic = create_critic(args)
    logger.info("Using critic: %s", type(critic).__name__)
    logger.info("Using tool preset: %s", args.tool_preset)

    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_keep_first=args.condenser_keep_first,
    )

    evaluator = SWEBenchProEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
