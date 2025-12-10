#!/usr/bin/env python3
"""
Stage 2: Extract workflows from each cluster
- Type 1: General strategies (high-level guidance)
- Type 2: Specific workflows (concrete tool call examples)
- Both types should include reasoning and may have conditionals
"""

import json
import os
import time
from typing import Any, Dict, List

import litellm


def load_clusters(cluster_file: str) -> Dict[str, Any]:
    """Load cluster information."""
    with open(cluster_file, "r") as f:
        return json.load(f)


def load_trajectory_text(
    trajectory_dir: str, instance_id: str, max_tokens: int = 8000
) -> tuple[str, bool]:
    """
    Load cleaned trajectory text.
    Returns: (text, is_oversized)
    - text: the trajectory content (or None if file doesn't exist)
    - is_oversized: True if file exceeds max_tokens
    """
    file_path = os.path.join(trajectory_dir, f"{instance_id}.txt")
    if not os.path.exists(file_path):
        return None, False

    with open(file_path, "r") as f:
        text = f.read()

    # Rough token estimation (1 token ≈ 4 characters)
    estimated_tokens = len(text) / 4
    is_oversized = estimated_tokens > max_tokens

    return text, is_oversized


def extract_workflows_for_cluster(
    cluster_name: str,
    cluster_data: Dict[str, Any],
    trajectory_dir: str,
    model: str,
    base_url: str = None,
) -> tuple[Dict[str, List[Dict[str, str]]], List[str], List[str]]:
    """
    Extract workflows for a single cluster.
    Returns: (workflows, oversized_trajectories, missing_trajectories)
    """

    # Load all trajectories for this cluster
    trajectories = []
    oversized_trajectories = []
    missing_trajectories = []

    for instance_id in cluster_data["instance_ids"]:
        traj_text, is_oversized = load_trajectory_text(trajectory_dir, instance_id)

        if traj_text is None:
            missing_trajectories.append(instance_id)
            continue

        if is_oversized:
            oversized_trajectories.append(instance_id)
            print(
                f"  Warning: Trajectory {instance_id} is too large (>8k tokens), skipping"
            )
            continue

        trajectories.append({"instance_id": instance_id, "trajectory": traj_text})

    if not trajectories:
        print(f"Warning: No valid trajectories found for cluster {cluster_name}")
        return (
            {"type1_workflows": [], "type2_workflows": []},
            oversized_trajectories,
            missing_trajectories,
        )

    # Combine trajectories (limit to prevent token overflow)
    combined_text = ""
    for i, traj in enumerate(trajectories[:10]):  # Limit to 10 trajectories
        combined_text += (
            f"\n{'=' * 80}\nTRAJECTORY {i + 1}: {traj['instance_id']}\n{'=' * 80}\n"
        )
        combined_text += traj["trajectory"] + "\n"

    prompt = f"""You are analyzing Django code agent trajectories to extract reusable workflows.

IMPORTANT: The workflow can be generally applicable to all repositories.
IMPORTANT: The workflow can also be DJANGO repository specific workflows.

CLUSTER: {cluster_name}
DESCRIPTION: {cluster_data["description"]}

TRAJECTORIES:
{combined_text}

Your task is to extract TWO types of workflows from these Django repository trajectories:

TYPE 1 WORKFLOWS - General Strategies:
- High-level guidance for broader goals in Django development
- Step-by-step approach WITHOUT separate reasoning/action format
- Each step describes what to do following the previous step
- Examples:
  * "Fixing Django model ordering issues"
    - Step 1: Examine the model definition and Meta.ordering to understand the expected behavior
    - Step 2: Run a test query to confirm the issue and observe the generated SQL
    - Step 3: Search for the query generation code in django/db/models/sql/ directory
  * "Debugging Django autoreloader errors"
    - Step 1: Review the error stack trace to identify the failing module and line number
    - Step 2: Locate the relevant code in django/utils/autoreload.py
    - Step 3: Add error handling or path validation to prevent the issue
- Focus on the overall approach and decision-making patterns
- Can include conditionals (e.g., "If stack trace available, start there; otherwise search for error message")

TYPE 2 WORKFLOWS - Specific Action Workflows:
- Concrete tool call examples for common sub-tasks in a fixing process
- Contains specific commands, actions, and reasoning for each action
- Abstract non-fixed elements with {{variable_names}}
- Examples:
  * "Locating Django model field implementation"
    - Reasoning: Need to find where a specific field type is implemented to understand its behavior
    - Action: grep -r "class {{FieldType}}" django/db/models/fields/
  * "Running Django test suite for specific app"
    - Reasoning: Need to verify the fix works and doesn't break existing functionality
    - Action: cd tests && python runtests.py {{app_name}}
- Should be actionable and include specific tool usage patterns
- Can include conditionals based on observed patterns

You can see the difference between TYPE 1 and TYPE 2 workflows is TYPE 2 workflow can be a guide for one step in TYPE 1 workflow, but not vice versa.

REQUIREMENTS:
1. TYPE 1: Use simple step descriptions (Step 1, Step 2, etc.) - NO reasoning/action split
2. TYPE 2: Each step must include REASONING and specific ACTION
3. Track which trajectories contribute to each workflow
4. Make workflows concise and elegant - combine similar patterns
5. Don't create redundant workflows - each should be distinct

EXAMPLE FORMAT:

## Type 1: Debugging Django Query Generation Issues (General Strategy)
This workflow involves identifying and fixing issues in how Django generates SQL queries.

Step 1: Examine the model definition to understand the expected query behavior and identify any Meta options that might affect query generation
Step 2: Run a test query in Django shell or create a minimal test case to reproduce the issue and observe the generated SQL
Step 3: Search for query generation code in django/db/models/sql/ directory, focusing on files like query.py, compiler.py
Step 4: Add print statements or use debugger to trace how the query is being constructed
Step 5: Implement the fix and run tests to verify it resolves the issue without breaking existing functionality

## Type 2: Running Django Tests for Specific Module (Specific Workflow)
This workflow is used to run Django's test suite for a specific module to verify fixes or check for regressions.

Reasoning: Need to run tests for a specific Django module to verify a fix or check for regressions. Django has a custom test runner in tests/runtests.py that should be used.
Action: cd tests && python runtests.py {{module_name}} --settings={{settings_file}} -v 2

Now extract workflows from the provided trajectories. For each workflow, indicate which trajectory instance_ids contributed to it. Output in this JSON format:

{{
  "type1_workflows": [
    {{
      "name": "Workflow name",
      "description": "What this workflow accomplishes",
      "steps": [
        "Step 1: Description of what to do",
        "Step 2: Description of what to do next",
        ...
      ],
      "source_trajectories": ["instance_id1", "instance_id2", ...]
    }},
    ...
  ],
  "type2_workflows": [
    {{
      "name": "Workflow name",
      "description": "What this workflow accomplishes",
      "steps": [
        {{
          "reasoning": "Why this step is needed",
          "action": "Specific command/tool call"
        }},
        ...
      ],
      "source_trajectories": ["instance_id1", "instance_id2", ...]
    }},
    ...
  ]
}}

Output only the JSON, no other text."""

    # Configure litellm
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
    }
    if base_url:
        kwargs["api_base"] = base_url

    # Add retry logic to handle rate limiting and WAF blocks
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                print(f"  Error: {str(e)[:200]}")
                print(
                    f"  Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts")
                raise

    # Parse JSON from response
    response_text = response.choices[0].message.content.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    workflows = json.loads(response_text)
    return workflows, oversized_trajectories, missing_trajectories


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2: Extract workflows from clusters"
    )
    parser.add_argument("--cluster-file", required=True, help="JSON file with clusters")
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory with cleaned trajectory text files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for workflows"
    )
    parser.add_argument(
        "--model", default="gpt-5", help="Model to use (default: gpt-5)"
    )
    parser.add_argument(
        "--base-url",
        default="https://ai-gateway.andrew.cmu.edu/",
        help="API base URL (default: CMU AI Gateway, or set to None for official OpenAI API)",
    )
    parser.add_argument("--api-key", help="API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    # Setup
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required via --api-key or OPENAI_API_KEY env var")

    # Set API key for litellm
    os.environ["OPENAI_API_KEY"] = api_key

    print(f"Using model: {args.model}")
    if args.base_url:
        if "cmu.edu" in args.base_url:
            print("Using CMU AI Gateway")
        else:
            print(f"Base URL: {args.base_url}")
    else:
        print("Using official OpenAI API")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load clusters
    clusters = load_clusters(args.cluster_file)
    print(f"Loaded {len(clusters)} clusters")

    # Process each cluster
    all_workflows = {}
    problematic_clusters = {}  # Track clusters with issues
    cluster_list = list(clusters.items())

    # Create directory for problematic trajectories
    problem_dir = os.path.join(args.output_dir, "_problematic_trajectories")
    os.makedirs(problem_dir, exist_ok=True)

    for i, (cluster_name, cluster_data) in enumerate(cluster_list):
        print(f"\nProcessing cluster {i + 1}/{len(cluster_list)}: {cluster_name}")
        print(f"  Tasks: {len(cluster_data['instance_ids'])}")

        workflows, oversized, missing = extract_workflows_for_cluster(
            cluster_name, cluster_data, args.trajectory_dir, args.model, args.base_url
        )

        # Track problematic files
        if oversized or missing:
            problematic_clusters[cluster_name] = {
                "oversized_trajectories": oversized,
                "missing_trajectories": missing,
            }

            # Move oversized trajectory files to problem directory
            for instance_id in oversized:
                src = os.path.join(args.trajectory_dir, f"{instance_id}.txt")
                dst = os.path.join(problem_dir, f"{instance_id}.txt")
                if os.path.exists(src):
                    import shutil

                    shutil.copy2(src, dst)
                    print(f"  Moved oversized trajectory to: {dst}")

        all_workflows[cluster_name] = workflows

        # Save individual cluster workflows
        # Sanitize cluster name for file system (replace / and spaces)
        safe_cluster_name = cluster_name.replace("/", "_").replace(" ", "_").lower()
        cluster_file = os.path.join(
            args.output_dir, f"{safe_cluster_name}_workflows.json"
        )
        with open(cluster_file, "w") as f:
            json.dump(workflows, f, indent=2)

        print(f"  Type 1 workflows: {len(workflows['type1_workflows'])}")
        print(f"  Type 2 workflows: {len(workflows['type2_workflows'])}")
        if oversized:
            print(f"  Oversized trajectories: {len(oversized)}")
        if missing:
            print(f"  Missing trajectories: {len(missing)}")
        print(f"  Saved to: {cluster_file}")

    # Save all workflows combined
    combined_file = os.path.join(args.output_dir, "all_workflows.json")
    with open(combined_file, "w") as f:
        json.dump(all_workflows, f, indent=2)

    # Save problematic clusters report
    if problematic_clusters:
        problem_report_file = os.path.join(
            args.output_dir, "_problematic_clusters_report.json"
        )
        with open(problem_report_file, "w") as f:
            json.dump(problematic_clusters, f, indent=2)

        print(f"\n{'=' * 80}")
        print(
            f"WARNING: {len(problematic_clusters)} cluster(s) have problematic trajectories:"
        )
        print(f"{'=' * 80}")
        for cluster_name, issues in problematic_clusters.items():
            print(f"\n  {cluster_name}:")
            if issues["oversized_trajectories"]:
                print(f"    - Oversized: {issues['oversized_trajectories']}")
            if issues["missing_trajectories"]:
                print(f"    - Missing: {issues['missing_trajectories']}")
        print(f"\nProblematic trajectories moved to: {problem_dir}")
        print(f"Report saved to: {problem_report_file}")
        print(f"{'=' * 80}\n")

    print("\nStage 2 complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"Combined workflows saved to: {combined_file}")


if __name__ == "__main__":
    main()
