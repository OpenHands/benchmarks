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
        choices=["action_sequence", "problem_description", "code_modification", "random"],
        default="action_sequence",
        help="Clustering method (default: action_sequence)"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Clustering similarity threshold [0-1] (default: 0.5)"
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
    output_file = output_dir / "workflows.json"

    logger.info("=" * 60)
    logger.info("CAWM Pipeline Configuration")
    logger.info("=" * 60)
    logger.info(f"  Input:        {args.input}")
    logger.info(f"  Output:       {output_file}")
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
    pipeline = CAWMPipeline(llm_client=llm_client, config=config)

    logger.info("Running pipeline...")
    logger.info("  Step 1/3: Clustering trajectories...")
    # Note: Pipeline.run() does compress -> cluster -> induce internally

    workflows = pipeline.run(trajectories)

    logger.info(f"Pipeline complete. Extracted {len(workflows)} workflows.")

    # Save results
    logger.info(f"Saving workflows to {output_file}...")

    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "input_file": args.input,
            "num_trajectories": len(trajectories),
            "config": {
                "compression": args.compression,
                "clustering": args.clustering,
                "threshold": args.threshold,
                "level": args.level,
                "model": args.model,
            }
        },
        "workflows": [w.to_dict() for w in workflows],
        "count": len(workflows),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Also save a summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"CAWM Workflow Extraction Summary\n")
        f.write(f"{'=' * 40}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Trajectories processed: {len(trajectories)}\n")
        f.write(f"Workflows extracted: {len(workflows)}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Compression: {args.compression}\n")
        f.write(f"  Clustering: {args.clustering} (threshold={args.threshold})\n")
        f.write(f"  Level: {args.level}\n")
        f.write(f"  Model: {args.model}\n\n")
        f.write(f"Workflows:\n")
        f.write(f"{'-' * 40}\n")
        for i, wf in enumerate(workflows, 1):
            f.write(f"\n{i}. {wf.description}\n")
            f.write(f"   Category: {wf.category}\n")
            f.write(f"   Steps: {len(wf.steps)}\n")
            for j, step in enumerate(wf.steps, 1):
                f.write(f"     {j}. [{step.action_type}] {step.action[:60]}...\n" if len(step.action) > 60 else f"     {j}. [{step.action_type}] {step.action}\n")

    logger.info(f"Summary saved to {summary_file}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"SUCCESS: Extracted {len(workflows)} workflows")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")
    print("\nWorkflows:")
    for i, wf in enumerate(workflows, 1):
        print(f"  {i}. [{wf.category}] {wf.description[:50]}... ({len(wf.steps)} steps)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
