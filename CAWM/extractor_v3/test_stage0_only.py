#!/usr/bin/env python3
"""
Test Stage 0 without LLM - just extract and verify data structure
"""

import json
import os
from datetime import datetime


def extract_relevant_events(trajectory):
    """Extract problem description and action events."""
    # Get problem description from instruction field (extract from <issue_description> tag)
    instruction = trajectory.get("instruction", "")
    if "<issue_description>" in instruction:
        start_tag = "<issue_description>"
        end_tag = "</issue_description>"
        start = instruction.find(start_tag) + len(start_tag)
        end = instruction.find(end_tag)
        problem_description = instruction[start:end].strip()
    else:
        problem_description = instruction

    # Get history events
    events = trajectory.get("history", [])
    action_events = []

    # Extract ActionEvents with timing
    prev_timestamp = None
    for event in events:
        event_kind = event.get("kind", "")
        current_timestamp = event.get("timestamp")

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


def test_extraction():
    """Test extraction on first few trajectories."""

    input_file = (
        "/home/tsljgj/private/benchmarks/CAWM/trajectories/resolved_trajectories.jsonl"
    )
    output_dir = "/home/tsljgj/private/benchmarks/CAWM/extractor_v3_output/test_stage0"

    os.makedirs(output_dir, exist_ok=True)

    # Load and process first 3 trajectories
    count = 0
    max_count = 3
    problem_list = []

    with open(input_file, "r") as f:
        for line in f:
            if count >= max_count:
                break

            trajectory = json.loads(line.strip())
            instance_id = trajectory.get("instance_id")

            print(f"\n{'=' * 60}")
            print(f"Processing: {instance_id}")
            print("=" * 60)

            problem_desc, action_events = extract_relevant_events(trajectory)

            print(f"Problem description length: {len(problem_desc)} chars")
            print(f"Action events found: {len(action_events)}")

            if problem_desc and action_events:
                # Save problem description
                problem_list.append(
                    {"instance_id": instance_id, "problem_description": problem_desc}
                )

                # Save action events to JSON (instead of using LLM)
                output_file = os.path.join(output_dir, f"{instance_id}.json")
                with open(output_file, "w") as out:
                    json.dump(
                        {
                            "instance_id": instance_id,
                            "problem_description": problem_desc,
                            "action_events": action_events,
                        },
                        out,
                        indent=2,
                    )

                print(f"✅ Saved to: {output_file}")
                count += 1
            else:
                print("⚠️ Skipping - missing data")

    # Save problem list
    problem_list_file = os.path.join(output_dir, "problem_descriptions.json")
    with open(problem_list_file, "w") as f:
        json.dump(problem_list, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Test complete!")
    print(f"Processed {count} trajectories")
    print(f"Output: {output_dir}")
    print(f"Problem list: {problem_list_file}")
    print("=" * 60)


if __name__ == "__main__":
    test_extraction()
