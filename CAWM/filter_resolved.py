#!/usr/bin/env python3
"""
Extract resolved trajectories from SWE-bench evaluation output.

This script reads the evaluation results JSON file to get the list of resolved
instance IDs, then extracts the corresponding trajectories from output.jsonl
and writes them to a target JSONL file.

Usage:
    python filter_resolved.py <eval_output_dir> [target_jsonl]

Example:
    python /home/tsljgj/private/benchmarks/CAWM/filter_resolved.py \
        /home/tsljgj/private/benchmarks/eval_outputs/princeton-nlp__SWE-bench_Lite-test/openrouter/moonshotai/kimi-k2-thinking_sdk_e485bba_maxiter_100_N_initial
"""

import argparse
import json
import os
import sys


DEFAULT_TARGET = (
    "/home/tsljgj/private/benchmarks/CAWM/trajectories/resolved_trajectories.jsonl"
)


def load_resolved_ids(eval_output_dir: str) -> set:
    """
    Load the list of resolved instance IDs from the evaluation results JSON file.

    Args:
        eval_output_dir: Path to the evaluation output directory

    Returns:
        Set of resolved instance IDs
    """
    eval_results_file = os.path.join(
        eval_output_dir, "openhands.eval_output.swebench.json"
    )

    if not os.path.exists(eval_results_file):
        raise FileNotFoundError(
            f"Evaluation results file not found: {eval_results_file}"
        )

    with open(eval_results_file, "r") as f:
        eval_results = json.load(f)

    resolved_ids = eval_results.get("resolved_ids", [])
    print(f"Found {len(resolved_ids)} resolved instance IDs")

    return set(resolved_ids)


def extract_trajectories(
    eval_output_dir: str, resolved_ids: set, target_file: str
) -> int:
    """
    Extract trajectories for resolved instances and write to target file.

    Args:
        eval_output_dir: Path to the evaluation output directory
        resolved_ids: Set of resolved instance IDs to extract
        target_file: Path to the target JSONL file

    Returns:
        Number of trajectories written
    """
    output_jsonl = os.path.join(eval_output_dir, "output.jsonl")

    if not os.path.exists(output_jsonl):
        raise FileNotFoundError(f"Output JSONL file not found: {output_jsonl}")

    # Create target directory if it doesn't exist
    target_dir = os.path.dirname(target_file)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    written_count = 0
    found_ids = set()

    with open(output_jsonl, "r") as infile, open(target_file, "a") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                trajectory = json.loads(line)
                instance_id = trajectory.get("instance_id")

                if instance_id in resolved_ids:
                    outfile.write(line + "\n")
                    written_count += 1
                    found_ids.add(instance_id)
                    print(f"  Extracted: {instance_id}")

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)

    # Report any resolved IDs that weren't found in output.jsonl
    missing_ids = resolved_ids - found_ids
    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} resolved IDs not found in output.jsonl:")
        for mid in sorted(missing_ids):
            print(f"  - {mid}")

    return written_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract resolved trajectories from SWE-bench evaluation output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "eval_output_dir",
        help="Path to the evaluation output directory containing openhands.eval_output.swebench.json and output.jsonl",
    )
    parser.add_argument(
        "target_jsonl",
        nargs="?",
        default=DEFAULT_TARGET,
        help=f"Path to the target JSONL file (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target file instead of appending to it",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.eval_output_dir):
        print(
            f"Error: Evaluation output directory not found: {args.eval_output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Handle overwrite option
    if args.overwrite and os.path.exists(args.target_jsonl):
        os.remove(args.target_jsonl)
        print(f"Removed existing target file: {args.target_jsonl}")

    print(f"Evaluation output directory: {args.eval_output_dir}")
    print(f"Target file: {args.target_jsonl}")
    print()

    try:
        # Load resolved IDs
        resolved_ids = load_resolved_ids(args.eval_output_dir)

        if not resolved_ids:
            print("No resolved instances found. Nothing to extract.")
            sys.exit(0)

        # Extract trajectories
        print("\nExtracting resolved trajectories...")
        written_count = extract_trajectories(
            args.eval_output_dir, resolved_ids, args.target_jsonl
        )

        print(
            f"\nSuccessfully wrote {written_count} trajectories to {args.target_jsonl}"
        )

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
