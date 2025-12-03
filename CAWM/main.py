#!/usr/bin/env python3
"""
CAWM Main Entry Point

Run the full CAWM pipeline on trajectory data to extract workflows.

Usage:
    python CAWM/main.py --output workflow/2025-12-02
    python CAWM/main.py --output workflow/testing --compression key_step_extraction --clustering action_sequence
    python CAWM/main.py --help
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Ensure CAWM is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CAWM import (
    CAWMPipeline,
    PipelineConfig,
    LLMClient,
    Trajectory,
    CompressionStrategy,
    SimilarityMethod,
    WorkflowLevel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CAWM")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAWM - Extract workflows from agent trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default settings
  python CAWM/main.py --output workflow/experiment-1

  # Custom compression and clustering
  python CAWM/main.py --output workflow/test \\
      --compression hierarchical_summarization \\
      --clustering problem_description \\
      --threshold 0.3

  # Use specific workflow level
  python CAWM/main.py --output workflow/specific-workflows \\
      --level specific

  # Limit number of trajectories (for testing)
  python CAWM/main.py --output workflow/quick-test --limit 5
        """
    )

    # Required
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for workflows (e.g., workflow/2025-12-02)"
    )

    # Input
    parser.add_argument(
        "--input", "-i",
        default="CAWM/trajectories/resolved_trajectories.jsonl",
        help="Input trajectory JSONL file (default: CAWM/trajectories/resolved_trajectories.jsonl)"
    )

    # Compression
    parser.add_argument(
        "--compression", "-c",
        choices=["key_step_extraction", "hierarchical_summarization", "action_type_filtering", "no_op"],
        default="key_step_extraction",
        help="Compression strategy (default: key_step_extraction)"
    )

    # Clustering
    parser.add_argument(
        "--clustering", "-k",
        choices=["action_sequence", "problem_description", "code_modification", "repository", "random"],
        default="problem_description",
        help="Clustering method (default: problem_description)"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.2,
        help="Clustering similarity threshold [0-1] (default: 0.2)"
    )

    # Induction
    parser.add_argument(
        "--level", "-l",
        choices=["general", "specific"],
        default="general",
        help="Workflow abstraction level (default: general)"
    )

    # LLM
    parser.add_argument(
        "--model", "-m",
        default="moonshotai/kimi-k2-0905",
        help="LLM model to use via OpenRouter (default: moonshotai/kimi-k2-0905)"
    )

    parser.add_argument(
        "--provider", "-p",
        choices=["openrouter", "openai", "anthropic"],
        default="openrouter",
        help="LLM provider (default: openrouter)"
    )

    # Other
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of trajectories to process (for testing)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )

    return parser.parse_args()


def validate_api_key(provider: str) -> bool:
    """Check if required API key is set."""
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var and os.getenv(env_var):
        return True
    return False


def main():
    args = parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate API key
    if not validate_api_key(args.provider):
        logger.error(f"API key not found for provider '{args.provider}'")
        key_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        print(f"\nPlease set: export {key_map[args.provider]}='your-api-key'")
        sys.exit(1)

    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CAWM Pipeline Configuration")
    logger.info("=" * 60)
    logger.info(f"  Input:        {args.input}")
    logger.info(f"  Output:       {output_dir}/")
    logger.info(f"  Compression:  {args.compression}")
    logger.info(f"  Clustering:   {args.clustering} (threshold={args.threshold})")
    logger.info(f"  Level:        {args.level}")
    logger.info(f"  Model:        {args.model}")
    logger.info(f"  Provider:     {args.provider}")
    if args.limit:
        logger.info(f"  Limit:        {args.limit} trajectories")
    logger.info("=" * 60)

    # Load trajectories
    logger.info(f"Loading trajectories from {args.input}...")
    trajectories = Trajectory.load_from_jsonl(args.input)

    if not trajectories:
        logger.error("No trajectories loaded!")
        sys.exit(1)

    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Apply limit if specified
    if args.limit and args.limit < len(trajectories):
        trajectories = trajectories[:args.limit]
        logger.info(f"Limited to {len(trajectories)} trajectories")

    # Map string args to enums
    compression_map = {
        "key_step_extraction": CompressionStrategy.KEY_STEP_EXTRACTION,
        "hierarchical_summarization": CompressionStrategy.HIERARCHICAL_SUMMARIZATION,
        "action_type_filtering": CompressionStrategy.ACTION_TYPE_FILTERING,
        "no_op": CompressionStrategy.NO_OP,
    }

    clustering_map = {
        "action_sequence": SimilarityMethod.ACTION_SEQUENCE,
        "problem_description": SimilarityMethod.PROBLEM_DESCRIPTION,
        "code_modification": SimilarityMethod.CODE_MODIFICATION,
        "repository": SimilarityMethod.REPOSITORY,
        "random": SimilarityMethod.RANDOM,
    }

    level_map = {
        "general": WorkflowLevel.GENERAL,
        "specific": WorkflowLevel.SPECIFIC,
    }

    # Configure pipeline
    config = PipelineConfig(
        compression_strategy=compression_map[args.compression],
        clustering_method=clustering_map[args.clustering],
        clustering_threshold=args.threshold,
        workflow_level=level_map[args.level],
        llm_model=args.model,
    )

    # Initialize LLM client
    logger.info(f"Initializing LLM client ({args.provider})...")
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
    )

    # Create and run pipeline
    logger.info("Creating pipeline...")
    pipeline = CAWMPipeline(
        llm_client=llm_client,
        config=config,
        output_dir=str(output_dir)  # Pass output dir for intermediate state saving
    )

    logger.info("Running pipeline...")
    logger.info("  Step 1/3: Clustering trajectories...")
    logger.info("  (Intermediate states will be saved to output directory)")
    # Note: Pipeline.run() does compress -> cluster -> induce internally

    workflows = pipeline.run(trajectories)

    logger.info(f"Pipeline complete. Extracted {len(workflows)} experiences.")
    # Note: All output files (workflows.json, summary.txt, clusters.json,
    # induction_details.json, pipeline_stats.json) are written by the
    # pipeline's OutputManager during execution.

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"SUCCESS: Extracted {len(workflows)} experiences")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nOutput files (all created at start, updated progressively):")
    print(f"  Main outputs:")
    print(f"    - workflows.json")
    print(f"    - summary.txt")
    print(f"  Intermediate states (for debugging/analysis):")
    print(f"    - clusters.json (clustering details)")
    print(f"    - induction_details.json (per-cluster induction)")
    print(f"    - pipeline_stats.json (timing & statistics)")
    print("\nExperiences preview:")
    for i, wf in enumerate(workflows[:10], 1):
        print(f"  {i}. [{wf.category}] {wf.description[:60]}...")
    if len(workflows) > 10:
        print(f"  ... and {len(workflows) - 10} more")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
