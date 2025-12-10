#!/usr/bin/env python3
"""
Debug script for Stage 1: Trajectory Segmentation

This runs ONLY Stage 1 on a single trajectory to help debug.
"""

import json
from pathlib import Path

from extractor_openai import WorkflowExtractor


def debug_stage1():
    """Debug Stage 1 segmentation on a single trajectory"""

    print("=" * 70)
    print("DEBUG: Stage 1 - Trajectory Segmentation")
    print("=" * 70)
    print()

    # Configuration
    base_url = "https://ai-gateway.andrew.cmu.edu/"
    model = "gpt-5"  # Your key has access to: gpt-5, gpt-5-mini, claude-sonnet-4

    # Load the 8th trajectory from resolved_trajectories.jsonl
    trajectory_file = Path("./trajectories/resolved_trajectories.jsonl")
    trajectory_index = 7  # 8th trajectory (0-indexed)

    print(f"Loading trajectory #{trajectory_index + 1} from {trajectory_file}")

    # Read the specific line from JSONL
    with open(trajectory_file, "r") as f:
        for i, line in enumerate(f):
            if i == trajectory_index:
                trajectory_data = json.loads(line)
                break
        else:
            print(f"❌ Trajectory index {trajectory_index} not found in file!")
            return 1

    instance_id = trajectory_data["instance_id"]
    events = trajectory_data.get("history", [])

    print("Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  Instance ID: {instance_id}")
    print(f"  Events in trajectory: {len(events)}")
    print()

    # Create extractor
    print("Creating extractor...")
    extractor = WorkflowExtractor(
        model=model, output_dir="./debug_output", base_url=base_url
    )
    print("✅ Extractor created")
    print()

    # Load and inspect trajectory
    print("=" * 70)
    print("STEP 1: Load trajectory data")
    print("=" * 70)
    print(f"✅ Loaded {len(events)} events")
    print()

    # Show first 3 events
    print("First 3 events:")
    for i, event in enumerate(events[:3]):
        print(f"\nEvent {i}:")
        for key in ["timestamp", "role", "action"]:
            if key in event:
                value = str(event[key])[:100]
                print(f"  {key}: {value}")
    print()

    # Format for LLM
    print("=" * 70)
    print("STEP 2: Format trajectory for LLM")
    print("=" * 70)
    formatted = extractor.format_trajectory_for_segmentation(events)
    print(f"✅ Formatted trajectory: {len(formatted)} chars")
    print()
    print("First 500 chars of formatted trajectory:")
    print("-" * 70)
    print(formatted[:500])
    print("...")
    print()

    # Create prompt
    print("=" * 70)
    print("STEP 3: Create segmentation prompt")
    print("=" * 70)
    prompt = extractor.create_segmentation_prompt(instance_id, formatted, len(events))
    print(f"✅ Created prompt: {len(prompt)} chars")
    print()
    print("First 500 chars of prompt:")
    print("-" * 70)
    print(prompt[:500])
    print("...")
    print()

    # Call LLM
    print("=" * 70)
    print("STEP 4: Call LLM")
    print("=" * 70)
    print(f"Calling {model} via {base_url}")
    print("This may take 30-60 seconds...")
    print()

    try:
        raw_response = extractor.call_llm(prompt)
        print(f"✅ Received response: {len(raw_response)} chars")
        print()
        print("Response (first 1000 chars):")
        print("-" * 70)
        print(raw_response[:1000])
        print("...")
        print()

        # Parse JSON
        print("=" * 70)
        print("STEP 5: Parse JSON response")
        print("=" * 70)

        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            print("❌ No JSON found in response!")
            print()
            print("Full response:")
            print(raw_response)
            return 1

        json_str = raw_response[json_start:json_end]
        print(f"Extracted JSON: {len(json_str)} chars")
        print()

        result_data = json.loads(json_str)
        segments = result_data.get("segments", [])

        print(f"✅ Parsed {len(segments)} segments")
        print()

        # Show segments
        print("=" * 70)
        print("STEP 6: Examine segments")
        print("=" * 70)

        for seg in segments:
            print(f"\nSegment {seg['segment_id']}:")
            print(f"  Events: {seg['start_event']}-{seg['end_event']}")
            print(f"  Purpose: {seg['purpose']}")
            print(f"  Goal: {seg['goal'][:80]}...")
            print(f"  Outcome: {seg['outcome']}")
            print(f"  Boundary: {seg['boundary_reason']}")

        print()
        print("=" * 70)
        print("✅ Stage 1 Debug Complete!")
        print("=" * 70)
        print()
        print(f"Total segments extracted: {len(segments)}")
        print("Output saved to: ./debug_output/stage1_segments/")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(debug_stage1())
