import os
import sys


# Add project root to path
sys.path.append(os.getcwd())

import logging  # noqa: E402
from typing import List  # noqa: E402

from CAWM import (  # noqa: E402
    CAWMPipeline,
    CompressionStrategy,
    LLMClient,
    PipelineConfig,
    SimilarityMethod,
    Trajectory,
    WorkflowLevel,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_trajectories(path: str, n: int = 3) -> List[Trajectory]:
    """Load first n trajectories from file."""
    logger.info(f"Loading top {n} trajectories from {path}...")
    trajectories = []
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                if line.strip():
                    data = import_json(line)
                    trajectories.append(Trajectory.from_raw(data))
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)
    return trajectories


def import_json(line):
    import json

    return json.loads(line)


def main():
    # 1. Check API Key
    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        logger.error(
            "No API key found! Please set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
        )
        print("\nTo fix this, run one of:")
        print("  export OPENROUTER_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-...'")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # 2. Load Data
    input_path = "CAWM/trajectories/resolved_trajectories.jsonl"
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    trajectories = load_sample_trajectories(input_path, n=3)
    logger.info(f"Loaded {len(trajectories)} trajectories.")

    # 3. Configure Pipeline (Testing Stage 2 features)
    # We use HIERARCHICAL_SUMMARIZATION to test LLM compression
    # We use ACTION_SEQUENCE for clustering
    config = PipelineConfig(
        compression_strategy=CompressionStrategy.HIERARCHICAL_SUMMARIZATION,
        clustering_method=SimilarityMethod.ACTION_SEQUENCE,
        workflow_level=WorkflowLevel.GENERAL,
        clustering_threshold=0.3,  # Loose threshold to force some clustering on small sample
    )

    # 4. Initialize Client
    # Defaults to anthropic/claude-3.5-sonnet via OpenRouter if not specified,
    # but adapts based on available keys/provider.
    provider = "openrouter"
    if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        provider = "anthropic"
    elif os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        provider = "openai"
        config.llm_model = "gpt-4o"  # Switch model if using OpenAI direct

    logger.info(f"Initializing Pipeline with provider={provider}...")
    llm_client = LLMClient(provider=provider)

    pipeline = CAWMPipeline(llm_client=llm_client, config=config)

    # 5. Run
    logger.info("Running CAWM Pipeline...")
    workflows = pipeline.run(trajectories)

    # 6. Output Results
    print("\n" + "=" * 50)
    print(f"Successfully Induced {len(workflows)} Workflows")
    print("=" * 50)

    for i, wf in enumerate(workflows, 1):
        print(f"\nWorkflow {i}: {wf.description}")
        print(f"Category: {wf.category}")
        print(f"Steps ({len(wf.steps)}):")
        for step in wf.steps:
            print(f"  - [{step.action_type}] {step.action} (Reason: {step.reasoning})")


if __name__ == "__main__":
    main()
