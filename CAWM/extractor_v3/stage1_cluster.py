#!/usr/bin/env python3
"""
Stage 1: Cluster problems based on similarity
- Group similar tasks together (similar task types, similar errors, etc.)
- Keep clustering simple and broad
- Each group should have no more than 10 tasks
"""

import json
import os
from typing import Dict, List

import litellm


def load_problem_descriptions(problem_file: str) -> List[Dict[str, str]]:
    """Load problem descriptions from JSON file."""
    with open(problem_file, "r") as f:
        return json.load(f)


def cluster_problems_with_llm(
    problems: List[Dict[str, str]], model: str, base_url: str = None
) -> Dict[str, List[str]]:
    """Use LLM to cluster problems into groups."""

    # Prepare problem list
    problem_list = []
    for i, prob in enumerate(problems):
        problem_list.append(
            f"{i}. {prob['instance_id']}: {prob['problem_description'][:200]}..."
        )

    problems_text = "\n".join(problem_list)

    prompt = f"""You are helping to cluster coding task problems based on similarity.

TASK: Group these {len(problems)} problems into clusters based on:
- Similar task types (e.g., bug fixing, feature addition, refactoring)
- Similar error types (e.g., algorithm errors, exception handling, API issues)
- Similar domains (e.g., web frameworks, data processing, testing)

RULES:
1. Keep clustering simple and broad - focus on high-level similarity
2. Each group should have no more than 10 tasks
3. If some tasks don't fit well with others, put them in a "miscellaneous" group
4. Give each cluster a clear, descriptive name
5. It is allowed one task to belong to multiple clusters if it fits well in both (but must fit in well enough)
6. One task cannot belong to more than two clusters

PROBLEMS:
{problems_text}

Please output your clustering in this JSON format:
{{
  "clusters": [
    {{
      "cluster_name": "Algorithm Bug Fixes",
      "description": "Tasks involving fixing algorithmic errors in core logic",
      "instance_ids": ["django__django-11283", "sympy__sympy-12171", ...]
    }},
    ...
  ]
}}

Output only the JSON, no other text."""

    # Configure litellm
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
    }
    if base_url:
        kwargs["api_base"] = base_url

    response = litellm.completion(**kwargs)

    # Parse JSON from response
    response_text = response.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    clustering = json.loads(response_text)

    # Convert to dict mapping cluster_name -> list of instance_ids
    result = {}
    for cluster in clustering["clusters"]:
        result[cluster["cluster_name"]] = {
            "description": cluster["description"],
            "instance_ids": cluster["instance_ids"],
        }

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Cluster problems")
    parser.add_argument(
        "--problem-file", required=True, help="JSON file with problem descriptions"
    )
    parser.add_argument(
        "--output-file", required=True, help="Output JSON file for clusters"
    )
    parser.add_argument(
        "--model", default="gpt-5", help="Model to use (default: gpt-5)"
    )
    parser.add_argument(
        "--base-url",
        default="https://ai-gateway.andrew.cmu.edu/",
        help="API base URL (default: CMU AI Gateway)",
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
    print(f"Base URL: {args.base_url}")

    # Load problems
    problems = load_problem_descriptions(args.problem_file)
    print(f"Loaded {len(problems)} problem descriptions")

    # Cluster problems
    print("Clustering problems with LLM...")
    clusters = cluster_problems_with_llm(problems, args.model, args.base_url)

    # Save clusters
    with open(args.output_file, "w") as f:
        json.dump(clusters, f, indent=2)

    print("\nStage 1 complete!")
    print(f"Created {len(clusters)} clusters:")
    for name, data in clusters.items():
        print(f"  - {name}: {len(data['instance_ids'])} tasks")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
