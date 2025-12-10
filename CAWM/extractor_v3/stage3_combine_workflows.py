#!/usr/bin/env python3
"""
Stage 3: Combine similar workflows across all clusters
- Merge workflows that are talking about the same thing
- Ensure no redundancy in final workflow set
"""

import json
import os
from typing import Any, Dict, List

import litellm


def load_all_workflows(workflow_file: str) -> Dict[str, Any]:
    """Load all workflows from combined file."""
    with open(workflow_file, "r") as f:
        return json.load(f)


def parse_llm_response(response_text: str) -> dict:
    """Parse LLM response with robust handling of markdown code blocks."""
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from LLM")

    response_text = response_text.strip()

    # Handle markdown code blocks
    if response_text.startswith("```"):
        # Find all code blocks
        parts = response_text.split("```")
        # Usually the JSON is in the second part (index 1)
        if len(parts) >= 2:
            response_text = parts[1]
            # Remove language identifier (e.g., 'json')
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

    # Try to parse JSON
    return json.loads(response_text)


def call_llm_with_retry(
    prompt: str, model: str, base_url: str = None, max_retries: int = 3
) -> dict:
    """Call LLM with retry logic and error handling."""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16000,
            }
            if base_url:
                kwargs["api_base"] = base_url

            response = litellm.completion(**kwargs)
            response_text = response.choices[0].message.content

            # Parse the response
            return parse_llm_response(response_text)

        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parsing error on attempt {attempt + 1}/{max_retries}")
            print(f"Error: {e}")
            print("\nRaw response (first 500 chars):")
            print("=" * 80)
            print(response_text[:500] if response_text else "(empty)")
            print("=" * 80)

            if attempt == max_retries - 1:
                # Save the problematic response
                debug_file = f"debug_llm_response_error_{int(os.times()[4])}.txt"
                with open(debug_file, "w") as f:
                    f.write("PROMPT:\n")
                    f.write("=" * 80 + "\n")
                    f.write(prompt + "\n\n")
                    f.write("RESPONSE:\n")
                    f.write("=" * 80 + "\n")
                    f.write(response_text if response_text else "(empty)")
                print(f"\n❌ Saved debug info to: {debug_file}")
                raise

            print(f"Retrying... ({attempt + 2}/{max_retries})")

        except Exception as e:
            print(f"\n⚠️  Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
            print(f"Retrying... ({attempt + 2}/{max_retries})")

    raise RuntimeError(f"Failed after {max_retries} attempts")


def batch_workflows(workflows: List[Dict], batch_size: int = 15) -> List[List[Dict]]:
    """Split workflows into batches for processing."""
    batches = []
    for i in range(0, len(workflows), batch_size):
        batches.append(workflows[i : i + batch_size])
    return batches


def combine_workflow_batch(
    workflows: List[Dict], workflow_type: str, model: str, base_url: str = None
) -> List[Dict]:
    """Combine a batch of workflows using LLM."""

    workflows_json = json.dumps(workflows, indent=2)

    if workflow_type == "type1":
        prompt = f"""You are combining similar workflows to eliminate redundancy.

TYPE 1 WORKFLOWS (General Strategies):
{workflows_json}

Your task:
1. Identify workflows that are talking about the same thing
2. Merge similar workflows into a single, more comprehensive workflow
3. Keep distinct workflows separate
4. Ensure no two workflows in the output are redundant
5. Preserve the reasoning and action details from the original workflows
6. Make the combined workflows more general and elegant

Output the combined workflows in this JSON format:
{{
  "workflows": [
    {{
      "name": "Combined workflow name",
      "description": "What this workflow accomplishes",
      "steps": [
        "Step 1: Description of what to do",
        "Step 2: Description of what to do next",
        ...
      ],
      "source_clusters": ["cluster1", "cluster2", ...],
      "source_trajectories": ["instance_id1", "instance_id2", ...]
    }},
    ...
  ]
}}

IMPORTANT: When merging workflows, combine their source_trajectories lists (all unique instance_ids from all merged workflows).

Output only the JSON, no other text."""

    else:  # type2
        prompt = f"""You are combining similar workflows to eliminate redundancy.

TYPE 2 WORKFLOWS (Specific Workflows):
{workflows_json}

Your task:
1. Identify workflows that are talking about the same thing
2. Merge similar workflows into a single, more comprehensive workflow
3. Keep distinct workflows separate
4. Ensure no two workflows in the output are redundant
5. Preserve specific tool calls and concrete examples
6. Make the combined workflows more general and elegant

Output the combined workflows in this JSON format:
{{
  "workflows": [
    {{
      "name": "Combined workflow name",
      "description": "What this workflow accomplishes",
      "steps": [
        {{
          "reasoning": "Why this step is needed",
          "action": "Specific command/tool call"
        }},
        ...
      ],
      "source_clusters": ["cluster1", "cluster2", ...],
      "source_trajectories": ["instance_id1", "instance_id2", ...]
    }},
    ...
  ]
}}

IMPORTANT: When merging workflows, combine their source_trajectories lists (all unique instance_ids from all merged workflows).

Output only the JSON, no other text."""

    result = call_llm_with_retry(prompt, model, base_url)
    return result["workflows"]


def combine_workflows_with_llm(
    all_workflows: Dict[str, Any],
    model: str,
    base_url: str = None,
    batch_size: int = 15,
) -> Dict[str, List[Dict]]:
    """Use LLM to combine similar workflows across clusters."""

    # Flatten all workflows into lists
    all_type1 = []
    all_type2 = []

    for cluster_name, workflows in all_workflows.items():
        for wf in workflows.get("type1_workflows", []):
            wf_copy = wf.copy()
            wf_copy["source_cluster"] = cluster_name
            all_type1.append(wf_copy)

        for wf in workflows.get("type2_workflows", []):
            wf_copy = wf.copy()
            wf_copy["source_cluster"] = cluster_name
            all_type2.append(wf_copy)

    print("\n📊 Total workflows to combine:")
    print(f"   Type 1: {len(all_type1)} workflows")
    print(f"   Type 2: {len(all_type2)} workflows")
    print(f"   Batch size: {batch_size} workflows per batch")

    # Process Type 1 workflows in batches
    type1_combined = []
    if all_type1:
        print("\n📝 Processing Type 1 workflows...")
        type1_batches = batch_workflows(all_type1, batch_size)
        print(f"   Split into {len(type1_batches)} batches")

        # First pass: combine within each batch
        batch_results = []
        for i, batch in enumerate(type1_batches, 1):
            print(
                f"   Batch {i}/{len(type1_batches)}: {len(batch)} workflows...", end=" "
            )
            combined = combine_workflow_batch(batch, "type1", model, base_url)
            batch_results.extend(combined)
            print(f"→ {len(combined)} workflows")

        print(f"   After batch processing: {len(batch_results)} workflows")

        # Second pass: combine across batches if needed
        if len(batch_results) > batch_size:
            print(
                f"   Final pass: combining {len(batch_results)} workflows across batches..."
            )
            final_batches = batch_workflows(batch_results, batch_size)
            type1_combined = []
            for i, batch in enumerate(final_batches, 1):
                print(
                    f"   Final batch {i}/{len(final_batches)}: {len(batch)} workflows...",
                    end=" ",
                )
                combined = combine_workflow_batch(batch, "type1", model, base_url)
                type1_combined.extend(combined)
                print(f"→ {len(combined)} workflows")
        else:
            type1_combined = batch_results

        print(f"✅ Type 1 final: {len(type1_combined)} workflows")

    # Process Type 2 workflows in batches
    type2_combined = []
    if all_type2:
        print("\n📝 Processing Type 2 workflows...")
        type2_batches = batch_workflows(all_type2, batch_size)
        print(f"   Split into {len(type2_batches)} batches")

        # First pass: combine within each batch
        batch_results = []
        for i, batch in enumerate(type2_batches, 1):
            print(
                f"   Batch {i}/{len(type2_batches)}: {len(batch)} workflows...", end=" "
            )
            combined = combine_workflow_batch(batch, "type2", model, base_url)
            batch_results.extend(combined)
            print(f"→ {len(combined)} workflows")

        print(f"   After batch processing: {len(batch_results)} workflows")

        # Second pass: combine across batches if needed
        if len(batch_results) > batch_size:
            print(
                f"   Final pass: combining {len(batch_results)} workflows across batches..."
            )
            final_batches = batch_workflows(batch_results, batch_size)
            type2_combined = []
            for i, batch in enumerate(final_batches, 1):
                print(
                    f"   Final batch {i}/{len(final_batches)}: {len(batch)} workflows...",
                    end=" ",
                )
                combined = combine_workflow_batch(batch, "type2", model, base_url)
                type2_combined.extend(combined)
                print(f"→ {len(combined)} workflows")
        else:
            type2_combined = batch_results

        print(f"✅ Type 2 final: {len(type2_combined)} workflows")

    return {"type1_workflows": type1_combined, "type2_workflows": type2_combined}


def filter_multi_source_workflows(
    workflows: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """Filter to keep only workflows with multiple source trajectories."""
    filtered = {"type1_workflows": [], "type2_workflows": []}

    for wf in workflows.get("type1_workflows", []):
        if len(wf.get("source_trajectories", [])) > 1:
            filtered["type1_workflows"].append(wf)

    for wf in workflows.get("type2_workflows", []):
        if len(wf.get("source_trajectories", [])) > 1:
            filtered["type2_workflows"].append(wf)

    return filtered


def convert_to_plain_text(workflows: Dict[str, List[Dict]]) -> str:
    """Convert workflows to plain text format."""
    lines = []

    # Type 1 Workflows
    lines.append("=" * 80)
    lines.append("TYPE 1 WORKFLOWS - General Strategies")
    lines.append("=" * 80)
    lines.append("")

    for wf in workflows.get("type1_workflows", []):
        name = wf.get("name", "Unnamed Workflow")
        description = wf.get("description", "")
        steps = wf.get("steps", [])
        sources = wf.get("source_trajectories", [])

        lines.append(f"## {name}")
        lines.append(f"{description}")
        lines.append(f"(Sources: {len(sources)} trajectories)")
        lines.append("")

        for step in steps:
            lines.append(step)

        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    # Type 2 Workflows
    lines.append("")
    lines.append("=" * 80)
    lines.append("TYPE 2 WORKFLOWS - Specific Action Workflows")
    lines.append("=" * 80)
    lines.append("")

    for wf in workflows.get("type2_workflows", []):
        name = wf.get("name", "Unnamed Workflow")
        description = wf.get("description", "")
        steps = wf.get("steps", [])
        sources = wf.get("source_trajectories", [])

        lines.append(f"## {name}")
        lines.append(f"{description}")
        lines.append(f"(Sources: {len(sources)} trajectories)")
        lines.append("")

        for step in steps:
            reasoning = step.get("reasoning", "")
            action = step.get("action", "")
            lines.append(f"Reasoning: {reasoning}")
            lines.append(f"Action: {action}")
            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: Combine similar workflows")
    parser.add_argument(
        "--workflow-file", required=True, help="JSON file with all workflows"
    )
    parser.add_argument(
        "--output-file", required=True, help="Output JSON file for combined workflows"
    )
    parser.add_argument(
        "--model", default="gpt-5", help="Model to use (default: gpt-5)"
    )
    parser.add_argument(
        "--base-url",
        default="https://ai-gateway.andrew.cmu.edu/",
        help="API base URL (default: CMU AI Gateway, use empty string for official OpenAI API)",
    )
    parser.add_argument("--api-key", help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Number of workflows to process per batch (default: 15)",
    )

    args = parser.parse_args()

    # Setup
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required via --api-key or OPENAI_API_KEY env var")

    # Set API key for litellm
    os.environ["OPENAI_API_KEY"] = api_key

    # Handle base_url - empty string means use official OpenAI API
    base_url = args.base_url if args.base_url else None

    print(f"Using model: {args.model}")
    if base_url:
        if "cmu.edu" in base_url:
            print("Using CMU AI Gateway")
        else:
            print(f"Base URL: {base_url}")
    else:
        print("Using official OpenAI API")

    # Load all workflows
    all_workflows = load_all_workflows(args.workflow_file)
    print(f"Loaded workflows from {len(all_workflows)} clusters")

    # Combine workflows
    print("\n" + "=" * 80)
    print("COMBINING WORKFLOWS")
    print("=" * 80)
    combined_workflows = combine_workflows_with_llm(
        all_workflows, args.model, base_url, args.batch_size
    )

    # Save unfiltered combined workflows
    unfiltered_file = args.output_file.replace(".json", "_unfiltered.json")
    with open(unfiltered_file, "w") as f:
        json.dump(combined_workflows, f, indent=2)

    print("\nUnfiltered workflows:")
    print(
        f"  Type 1 (General Strategies): {len(combined_workflows['type1_workflows'])} workflows"
    )
    print(
        f"  Type 2 (Specific Workflows): {len(combined_workflows['type2_workflows'])} workflows"
    )
    print(f"  Saved to: {unfiltered_file}")

    # Filter workflows with only one source trajectory
    print("\nFiltering workflows with multiple source trajectories...")
    filtered_workflows = filter_multi_source_workflows(combined_workflows)

    # Save filtered workflows (JSON)
    with open(args.output_file, "w") as f:
        json.dump(filtered_workflows, f, indent=2)

    print("\nFiltered workflows (multi-source only):")
    print(
        f"  Type 1 (General Strategies): {len(filtered_workflows['type1_workflows'])} workflows"
    )
    print(
        f"  Type 2 (Specific Workflows): {len(filtered_workflows['type2_workflows'])} workflows"
    )
    print(f"  Saved to: {args.output_file}")

    # Convert filtered workflows to plain text
    print("\nConverting filtered workflows to plain text...")
    plain_text = convert_to_plain_text(filtered_workflows)
    plain_text_file = args.output_file.replace(".json", ".txt")
    with open(plain_text_file, "w") as f:
        f.write(plain_text)

    print(f"  Plain text version saved to: {plain_text_file}")

    print("\nStage 3 complete!")
    print("\nOutput files:")
    print(f"  1. Unfiltered (JSON): {unfiltered_file}")
    print(f"  2. Filtered (JSON): {args.output_file}")
    print(f"  3. Filtered (Plain Text): {plain_text_file}")


if __name__ == "__main__":
    main()
