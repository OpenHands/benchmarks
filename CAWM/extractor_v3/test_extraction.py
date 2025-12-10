#!/usr/bin/env python3
"""
Test script to verify trajectory extraction without LLM calls
"""

import json


def test_trajectory_structure():
    """Test that we can extract problem description and action events correctly."""

    # Load first trajectory
    with open(
        "/home/tsljgj/private/benchmarks/CAWM/trajectories/resolved_trajectories.jsonl",
        "r",
    ) as f:
        first_line = f.readline()
        trajectory = json.loads(first_line)

    instance_id = trajectory.get("instance_id")
    instruction = trajectory.get("instruction", "")
    history = trajectory.get("history", [])

    # Count event types
    event_counts = {}
    action_events = []

    for event in history:
        kind = event.get("kind", "Unknown")
        event_counts[kind] = event_counts.get(kind, 0) + 1

        if kind == "ActionEvent":
            action_events.append(
                {
                    "thought": event.get("thought", ""),
                    "reasoning_content": event.get("reasoning_content", ""),
                    "action": event.get("action", {}),
                    "timestamp": event.get("timestamp"),
                }
            )

    # Print results
    print(f"Instance ID: {instance_id}")
    print(f"\nInstruction length: {len(instruction)} chars")
    print(f"Has <issue_description> tag: {'<issue_description>' in instruction}")
    print(f"\nTotal history events: {len(history)}")
    print("\nEvent type breakdown:")
    for kind, count in sorted(event_counts.items()):
        print(f"  {kind}: {count}")

    print(f"\nAction events found: {len(action_events)}")
    if action_events:
        print("\nFirst action event:")
        first_action = action_events[0]
        print(f"  Timestamp: {first_action['timestamp']}")
        print(f"  Has thought: {bool(first_action['thought'])}")
        print(f"  Has reasoning: {bool(first_action['reasoning_content'])}")
        print(f"  Action kind: {first_action['action'].get('kind', 'N/A')}")

    # Extract problem description from instruction
    if "<issue_description>" in instruction:
        start_tag = "<issue_description>"
        end_tag = "</issue_description>"
        start = instruction.find(start_tag) + len(start_tag)
        end = instruction.find(end_tag)
        problem_desc = instruction[start:end].strip()
        print(f"\nExtracted problem description length: {len(problem_desc)} chars")
        print(f"Problem description preview: {problem_desc[:200]}...")
    else:
        print("\nNo <issue_description> tag found")
        print(f"Instruction preview: {instruction[:200]}...")


if __name__ == "__main__":
    test_trajectory_structure()
