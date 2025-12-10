#!/usr/bin/env python3
"""
Stage 0: Clean trajectories
- Extract instruction (problem description) and ActionEvents only
- Convert to readable text format with action numbers and timing
- Generate problem description list
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import litellm


def load_trajectory_from_jsonl(file_path: str, instance_id: str) -> Dict[str, Any]:
    """Load a single trajectory from JSONL file by instance_id."""
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                if traj.get("instance_id") == instance_id:
                    return traj
    return None


def extract_relevant_events(
    trajectory: Dict[str, Any],
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Extract instruction (problem description) and ActionEvents.
    Returns: (problem_description, action_events_with_timing)
    """
    # Get problem description from instruction field (extract from <issue_description> tag)
    instruction = trajectory.get("instruction", "")
    if "<issue_description>" in instruction:
        start_tag = "<issue_description>"
        end_tag = "</issue_description>"
        start = instruction.find(start_tag) + len(start_tag)
        end = instruction.find(end_tag)
        problem_description = instruction[start:end].strip()
    else:
        # Fallback to full instruction if no tag found
        problem_description = instruction

    # Get history events
    events = trajectory.get("history", [])
    action_events = []

    # Extract ActionEvents with timing
    prev_timestamp = None
    for event in events:
        event_kind = event.get("kind", "")
        current_timestamp = event.get("timestamp")

        # Only extract ActionEvent (these contain actions)
        if event_kind == "ActionEvent":
            action_data = {
                "action": event.get("action", {}),
                "thought": event.get("thought", ""),
                "reasoning_content": event.get("reasoning_content", ""),
                "timestamp": current_timestamp,
            }

            # Calculate time since previous event
            if prev_timestamp and current_timestamp:
                try:
                    prev_dt = datetime.fromisoformat(
                        prev_timestamp.replace("Z", "+00:00")
                    )
                    curr_dt = datetime.fromisoformat(
                        current_timestamp.replace("Z", "+00:00")
                    )
                    time_diff = (curr_dt - prev_dt).total_seconds()
                    action_data["time_since_prev"] = time_diff
                except Exception:
                    action_data["time_since_prev"] = 0
            else:
                action_data["time_since_prev"] = 0

            action_events.append(action_data)
            prev_timestamp = current_timestamp

    return problem_description, action_events


def simplify_action_with_llm(
    action_events: List[Dict[str, Any]],
    problem_desc: str,
    model: str,
    base_url: str = None,
) -> str:
    """Use LLM to convert action events to concise, readable text."""

    # Prepare action events as JSON string
    actions_json = json.dumps(action_events, indent=2)

    prompt = f"""You are helping to convert a Django code agent's trajectory into readable text format.

PROBLEM DESCRIPTION:
{problem_desc}

ACTION EVENTS (JSON format):
{actions_json}

Please convert these action events into a clean, readable text format following these rules:

1. Start with the problem description clearly labeled
2. For each action, include:
   - Action number (Action 1, Action 2, etc.)
   - The agent's reasoning/thought (2-3 sentences MAXIMUM, be concise and elegant - capture the key logic and insight)
   - The tool call/action taken (preserve terminal commands, file paths, and concrete actions)

3. IMPORTANT - Handle code edits efficiently:
   - If the action contains a large code edit or file write, DO NOT include the entire code
   - Instead, summarize what was changed with a concrete example snippet (3-5 lines max)
   - Example: Instead of 500 lines of code, say "Modified the QuerySet.filter() method to handle negated ordering. Changed line 245 from `ordering.append(field)` to `ordering.append(('-' if desc else '') + field)`"
   - Keep grep/search results, but truncate if extremely long (first 5-10 matches are enough)

4. You should summarize the problem statement as well. Make sure you understand the problem correctly. If you are unsure, keep the original problem statement.

5. Use plain text, not JSON format

6. In brief, your task is to shorten the long action events into concise, elegant reasoning and actions, while preserving all important details.

Example output format:

PROBLEM: [problem description]

Action 1
Reasoning: The agent needs to understand the Django model structure to identify where the ordering issue occurs. Looking at the problem, it appears related to model inheritance and the Meta.ordering attribute. The agent should first locate the relevant Django model files to understand the current implementation and identify where the "-pk" ordering is being incorrectly converted to ASC instead of DESC in child models.
Action: grep -r "class.*Model" --include="*.py" django/db/models/

Action 2
Reasoning: After identifying potential model files, the agent needs to examine the specific implementation of model ordering in Django's query generation. This will help locate where the ordering direction is being lost during query construction for inherited models.
Action: cat django/db/models/sql/query.py | grep -A 20 "def get_ordering"

Now convert the provided trajectory:"""

    # Configure litellm
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
    }
    if base_url:
        kwargs["api_base"] = base_url

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


def process_trajectory(
    trajectory_file: str, output_dir: str, model: str, base_url: str, instance_id: str
) -> Dict[str, str]:
    """
    Process a single trajectory from JSONL file.
    Returns: dict with instance_id and problem_description
    """
    # Load trajectory from JSONL file
    trajectory = load_trajectory_from_jsonl(trajectory_file, instance_id)

    if not trajectory:
        print(f"Warning: Trajectory {instance_id} not found in {trajectory_file}")
        return None

    problem_desc, action_events = extract_relevant_events(trajectory)

    if not problem_desc:
        print(f"Warning: No problem description found for {instance_id}")
        return None

    if not action_events:
        print(f"Warning: No action events found for {instance_id}")
        return None

    # Convert to readable text using LLM
    try:
        readable_text = simplify_action_with_llm(
            action_events, problem_desc, model, base_url
        )

        # Validate the response is not empty
        if not readable_text or len(readable_text.strip()) < 50:
            print(
                f"Warning: LLM returned empty or very short response for {instance_id}, using fallback"
            )
            # Fallback: create simple text format without LLM
            readable_text = f"PROBLEM: {problem_desc}\n\n"
            for i, event in enumerate(action_events, 1):
                readable_text += (
                    f"Action {i} (Time: {event.get('time_since_prev', 0)}s)\n"
                )
                readable_text += f"Reasoning: {event.get('reasoning_content', event.get('thought', 'N/A'))}\n"
                readable_text += f"Action: {event.get('action', {})}\n\n"
    except Exception as e:
        print(f"Error: LLM call failed for {instance_id}: {str(e)}")
        print("  Using fallback format instead")
        # Fallback: create simple text format without LLM
        readable_text = f"PROBLEM: {problem_desc}\n\n"
        for i, event in enumerate(action_events, 1):
            readable_text += f"Action {i} (Time: {event.get('time_since_prev', 0)}s)\n"
            readable_text += f"Reasoning: {event.get('reasoning_content', event.get('thought', 'N/A'))}\n"
            readable_text += f"Action: {event.get('action', {})}\n\n"

    # Save to text file
    output_file = os.path.join(output_dir, f"{instance_id}.txt")
    with open(output_file, "w") as f:
        f.write(readable_text)

    print(
        f"Processed: {instance_id} ({len(action_events)} actions, {len(readable_text)} chars)"
    )

    return {"instance_id": instance_id, "problem_description": problem_desc}


def get_instance_ids_from_jsonl(file_path: str, limit: int = None) -> List[str]:
    """Get list of instance IDs from JSONL file."""
    instance_ids = []
    with open(file_path, "r") as f:
        for line in f:
            if limit and len(instance_ids) >= limit:
                break
            if line.strip():
                traj = json.loads(line)
                instance_id = traj.get("instance_id")
                if instance_id:
                    instance_ids.append(instance_id)
    return instance_ids


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 0: Clean trajectories")
    parser.add_argument(
        "--input-file", required=True, help="Input JSONL file with trajectories"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for cleaned text files"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of trajectories to process"
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using model: {args.model}")
    if args.base_url:
        if "cmu.edu" in args.base_url:
            print("Using CMU AI Gateway")
        else:
            print(f"Base URL: {args.base_url}")
    else:
        print("Using official OpenAI API")

    # Get instance IDs from file
    instance_ids = get_instance_ids_from_jsonl(args.input_file, args.limit)
    print(f"Found {len(instance_ids)} trajectories to process")

    # Process all trajectories
    problem_list = []
    for instance_id in instance_ids:
        result = process_trajectory(
            args.input_file, args.output_dir, args.model, args.base_url, instance_id
        )
        if result:
            problem_list.append(result)

    # Save problem description list
    problem_list_file = os.path.join(args.output_dir, "problem_descriptions.json")
    with open(problem_list_file, "w") as f:
        json.dump(problem_list, f, indent=2)

    print("\nStage 0 complete!")
    print(f"Processed {len(problem_list)} trajectories")
    print(f"Output saved to: {args.output_dir}")
    print(f"Problem list saved to: {problem_list_file}")


if __name__ == "__main__":
    main()
