#!/usr/bin/env python3
"""
LLM-First Workflow Extraction Pipeline - OpenAI Version

Uses OpenAI's o1 reasoning models for deep analysis.

Simple 3-stage pipeline:
1. Segmentation: Break trajectories into workflow segments
2. Clustering: Group similar segments at 3 levels (high/mid/low)
3. Extraction: Extract reusable workflows from clusters

Philosophy: Let LLMs do the heavy lifting, code only orchestrates.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class SegmentationResult:
    """Result from Stage 1: Segmentation"""

    instance_id: str
    total_events: int
    segments: List[Dict[str, Any]]
    raw_response: str


@dataclass
class ClusteringResult:
    """Result from Stage 2: Clustering"""

    level: str  # 'high', 'mid', or 'low'
    clusters: List[Dict[str, Any]]
    parent_cluster: Optional[str] = None
    raw_response: str = ""


@dataclass
class WorkflowResult:
    """Result from Stage 3: Workflow Extraction"""

    workflow: Dict[str, Any]
    cluster_name: str
    num_segments: int
    raw_response: str


class WorkflowExtractor:
    """
    LLM-First Workflow Extraction Pipeline using OpenAI

    Orchestrates LLM calls for segmentation, clustering, and extraction.
    Keeps code simple - LLM does the analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "o1",  # OpenAI o1 reasoning model
        output_dir: str = "./cawm_output_openai",
        base_url: Optional[str] = None,  # For custom endpoints like CMU AI Gateway
    ):
        # Initialize OpenAI client with optional base_url
        client_kwargs = {
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
            "max_retries": 3,
            "timeout": 300.0,  # 5 minutes for reasoning models
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        try:
            self.client = OpenAI(**client_kwargs)
        except TypeError as e:
            # Fallback: try without extra kwargs if there are compatibility issues
            print(f"⚠️  Warning: {e}")
            print("Retrying with minimal configuration...")
            client_kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories for each stage
        self.stage1_dir = self.output_dir / "stage1_segments"
        self.stage2_dir = self.output_dir / "stage2_clusters"
        self.stage3_dir = self.output_dir / "stage3_workflows"

        for d in [self.stage1_dir, self.stage2_dir, self.stage3_dir]:
            d.mkdir(exist_ok=True, parents=True)

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call OpenAI API with appropriate handling for o1 models

        o1 models don't use system prompts, so we prepend system context to user prompt
        """
        messages = []

        # For o1 models, combine system and user prompts
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        messages.append({"role": "user", "content": full_prompt})

        # Call API
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        return response.choices[0].message.content

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: TRAJECTORY SEGMENTATION
    # ═══════════════════════════════════════════════════════════════════

    def load_trajectory(self, trajectory_path: Path) -> List[Dict[str, Any]]:
        """Load trajectory from JSONL file"""
        events = []
        with open(trajectory_path, "r") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    def format_trajectory_for_segmentation(self, events: List[Dict[str, Any]]) -> str:
        """Format trajectory events into readable text for LLM"""
        from datetime import datetime

        lines = []

        # Determine if first timestamp is a string (ISO format) or number
        first_timestamp = events[0].get("timestamp", 0) if events else 0
        is_iso_format = isinstance(first_timestamp, str)
        base_time = None

        if is_iso_format and events:
            # Parse first event's timestamp as base
            base_time = datetime.fromisoformat(
                events[0].get("timestamp", "").replace("Z", "+00:00")
            )

        for i, event in enumerate(events):
            timestamp_raw = event.get("timestamp", 0)

            if is_iso_format:
                # Convert ISO string to seconds elapsed since first event
                if isinstance(timestamp_raw, str):
                    try:
                        event_time = datetime.fromisoformat(
                            timestamp_raw.replace("Z", "+00:00")
                        )
                        timestamp = (event_time - base_time).total_seconds()
                    except Exception:
                        timestamp = i * 15  # Fallback: 15 seconds per event
                else:
                    timestamp = i * 15
            else:
                timestamp = (
                    timestamp_raw if isinstance(timestamp_raw, (int, float)) else 0
                )

            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # Extract key information
            action_type = event.get("action", "Unknown")

            # Get reasoning (first 300 chars)
            reasoning = ""
            if "reasoning" in event:
                reasoning = str(event["reasoning"])[:300]
            elif "thought" in event:
                reasoning = str(event["thought"])[:300]

            # Get observation (first 150 chars)
            observation = ""
            if "observation" in event:
                obs = event["observation"]
                if isinstance(obs, dict):
                    obs = json.dumps(obs)
                observation = str(obs)[:150]

            # Format event
            lines.append(f"Event {i} ({time_str}):")
            lines.append(f"  Action: {action_type}")
            if reasoning:
                lines.append(f'  Reasoning: "{reasoning}"')
            if observation:
                lines.append(f'  Observation: "{observation}"')
            lines.append("")

        return "\n".join(lines)

    def create_segmentation_prompt(
        self, instance_id: str, formatted_trajectory: str, total_events: int
    ) -> str:
        """Create prompt for trajectory segmentation"""
        return f"""You are analyzing an agent trajectory to identify natural workflow segments.

A workflow segment is a cohesive sequence of actions where the agent is focused on accomplishing one specific sub-goal.

Here is the trajectory for instance {instance_id}:

{formatted_trajectory}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Divide this trajectory into workflow segments.

For each segment, identify:
1. Which events belong to it (start and end event numbers)
2. What is the PRIMARY PURPOSE of this segment?
   - UNDERSTANDING: Reading/analyzing code or output
   - LOCATING: Finding files, functions, or error locations
   - TESTING: Running tests and analyzing results
   - FIXING: Implementing changes or fixes
   - VERIFYING: Confirming the solution works

3. A brief GOAL (one sentence: what is the agent trying to accomplish?)

4. OUTCOME: Did this segment succeed, fail, or continue to next segment?

5. WHY did you end this segment here? (boundary reason)
   - time_gap: Agent paused to think
   - phase_change: Agent switched from one purpose to another
   - completion: Agent explicitly indicated completion
   - context_switch: Agent moved to different file/module

IMPORTANT RULES:
- Segments should be 3-10 events long (no single-event segments unless critical)
- Look for natural boundaries where agent's focus shifts
- When purpose changes (e.g., UNDERSTANDING → TESTING), that's a boundary
- When agent says "now I'll...", "next...", "done with..." - that's a boundary

OUTPUT FORMAT (JSON):
{{
  "instance_id": "{instance_id}",
  "total_events": {total_events},
  "segments": [
    {{
      "segment_id": 1,
      "start_event": 0,
      "end_event": 5,
      "duration_seconds": 92,
      "purpose": "UNDERSTANDING",
      "goal": "Read the issue description and understand the auto-reloader error",
      "outcome": "SUCCESS",
      "boundary_reason": "phase_change",
      "key_actions": ["read issue", "examine stack trace", "understand problem"]
    }},
    {{
      "segment_id": 2,
      ...
    }}
  ]
}}

Return ONLY the JSON, no other text."""

    def segment_trajectory(self, trajectory_path: Path) -> SegmentationResult:
        """
        Stage 1: Segment a single trajectory using LLM

        Returns:
            SegmentationResult with segments and metadata
        """
        instance_id = trajectory_path.stem

        print(f"\n{'=' * 70}")
        print(f"STAGE 1: Segmenting trajectory {instance_id}")
        print(f"{'=' * 70}")

        # Load trajectory
        events = self.load_trajectory(trajectory_path)
        print(f"Loaded {len(events)} events from {trajectory_path}")

        # Format for LLM
        formatted = self.format_trajectory_for_segmentation(events)

        # Create prompt
        prompt = self.create_segmentation_prompt(instance_id, formatted, len(events))

        print(f"Calling {self.model} for segmentation...")
        print(f"Prompt length: {len(prompt)} chars")

        # Call LLM
        raw_response = self.call_llm(prompt)

        print(f"Received response: {len(raw_response)} chars")

        # Parse response
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            json_str = raw_response[json_start:json_end]

            result_data = json.loads(json_str)
            segments = result_data["segments"]

            print(f"✅ Extracted {len(segments)} segments")

            # Print segment summary
            print("\nSegment Summary:")
            print("-" * 70)
            for seg in segments:
                print(f"  Segment {seg['segment_id']}: {seg['purpose']}")
                print(
                    f"    Events {seg['start_event']}-{seg['end_event']}: {seg['goal'][:60]}..."
                )
                print(
                    f"    Outcome: {seg['outcome']}, Boundary: {seg['boundary_reason']}"
                )

            result = SegmentationResult(
                instance_id=instance_id,
                total_events=len(events),
                segments=segments,
                raw_response=raw_response,
            )

            # Save to file
            output_file = self.stage1_dir / f"{instance_id}_segments.json"
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "instance_id": instance_id,
                        "total_events": len(events),
                        "segments": segments,
                        "raw_response": raw_response,
                    },
                    f,
                    indent=2,
                )
            print(f"\n💾 Saved to {output_file}")

            return result

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response as JSON: {e}")
            print(f"Response: {raw_response[:500]}...")
            raise

    def segment_all_trajectories(
        self, trajectory_dir: Path, limit: Optional[int] = None
    ) -> List[SegmentationResult]:
        """
        Segment all trajectories in a directory

        Args:
            trajectory_dir: Directory containing trajectory JSONL files
            limit: Optional limit on number of trajectories to process
        """
        trajectory_files = sorted(trajectory_dir.glob("*.jsonl"))

        if limit:
            trajectory_files = trajectory_files[:limit]

        print(f"\n{'=' * 70}")
        print("STAGE 1: TRAJECTORY SEGMENTATION")
        print(f"{'=' * 70}")
        print(f"Processing {len(trajectory_files)} trajectories from {trajectory_dir}")

        results = []
        for i, traj_path in enumerate(trajectory_files, 1):
            print(f"\n[{i}/{len(trajectory_files)}] Processing {traj_path.name}")
            try:
                result = self.segment_trajectory(traj_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {traj_path.name}: {e}")
                continue

        # Summary
        total_segments = sum(len(r.segments) for r in results)
        print(f"\n{'=' * 70}")
        print("STAGE 1 COMPLETE")
        print(f"{'=' * 70}")
        print(f"Processed: {len(results)} trajectories")
        print(f"Total segments: {total_segments}")
        if len(results) > 0:
            print(
                f"Average segments per trajectory: {total_segments / len(results):.1f}"
            )
        else:
            print("⚠️  No trajectories were successfully processed")

        # Save summary
        if len(results) > 0:
            summary_file = self.stage1_dir / "_summary.json"
            with open(summary_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "num_trajectories": len(results),
                        "total_segments": total_segments,
                        "avg_segments_per_trajectory": total_segments / len(results)
                        if len(results) > 0
                        else 0,
                        "trajectories": [
                            {
                                "instance_id": r.instance_id,
                                "total_events": r.total_events,
                                "num_segments": len(r.segments),
                            }
                            for r in results
                        ],
                    },
                    f,
                    indent=2,
                )

        return results

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2: MULTI-LEVEL CLUSTERING
    # ═══════════════════════════════════════════════════════════════════

    def create_segment_summary(self, segment: Dict[str, Any], instance_id: str) -> str:
        """Create compact summary of a segment for clustering"""
        seg_id = f"seg_{instance_id}_{segment['segment_id']}"
        return f"""Segment ID: {seg_id}
Instance: {instance_id}
Purpose: {segment["purpose"]}
Duration: {segment.get("duration_seconds", 0)}s
Goal: {segment["goal"]}
Key actions: {", ".join(segment.get("key_actions", []))}
Outcome: {segment["outcome"]}"""

    def create_clustering_prompt(
        self,
        segment_summaries: List[str],
        level: str,
        parent_cluster: Optional[str] = None,
    ) -> str:
        """Create prompt for clustering at specified level"""

        summaries_text = "\n\n".join(segment_summaries)

        if level == "high":
            return f"""You are analyzing {len(segment_summaries)} workflow segments from agent trajectories solving software engineering tasks.

Your task: Group these segments into HIGH-LEVEL STRATEGY clusters.

High-level strategies represent the overall approach or goal, such as:
- Bug Localization (finding where the bug is)
- Test Execution (running and analyzing tests)
- Iterative Debugging (test-fix-test cycles)
- Code Exploration (understanding the codebase)
- Solution Verification (confirming fix works)

Here are all segments:

{summaries_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Create 5-8 high-level strategy clusters.

For each cluster:
1. Give it a descriptive NAME
2. List which segment IDs belong to it
3. Explain WHY these segments belong together (what's the common strategy?)
4. How confident are you? (0.0-1.0)

OUTPUT (JSON):
{{
  "high_level_clusters": [
    {{
      "cluster_name": "Bug Localization Strategy",
      "segment_ids": ["seg_instance1_1", "seg_instance2_2", ...],
      "description": "Segments where agent is trying to find the location of a bug...",
      "confidence": 0.95,
      "num_segments": 85
    }}
  ]
}}

Return ONLY the JSON, no other text."""

        elif level == "mid":
            return f"""You previously identified a high-level cluster: "{parent_cluster}"
with {len(segment_summaries)} segments.

Now, subdivide this into MID-LEVEL PATTERN clusters.

Mid-level patterns are reusable approaches that could work across different frameworks/languages, such as:
- "Stack Trace to Code" (following stack trace to find bug location)
- "Grep-based Search" (searching codebase for patterns)
- "Progressive Exploration" (starting broad, narrowing down)

Here are the {len(segment_summaries)} segments in "{parent_cluster}":

{summaries_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Create 3-6 mid-level pattern clusters within this high-level cluster.

OUTPUT (JSON):
{{
  "parent_cluster": "{parent_cluster}",
  "mid_level_clusters": [
    {{
      "cluster_name": "Stack Trace Analysis Pattern",
      "segment_ids": [...],
      "description": "Agent follows stack trace to locate code...",
      "confidence": 0.88,
      "num_segments": 32
    }}
  ]
}}

Return ONLY the JSON, no other text."""

        else:  # low
            return f"""You previously identified:
- High-level: (parent of parent)
- Mid-level: "{parent_cluster}" ({len(segment_summaries)} segments)

Now, subdivide into LOW-LEVEL TASK-SPECIFIC clusters.

Low-level clusters are specific to a framework, language, or type of task:
- "Django Stack Trace Parsing" (Django-specific)
- "Flask Error Tracing" (Flask-specific)
- "Python Exception Tracking" (Python-general)

Here are the {len(segment_summaries)} segments:

{summaries_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Create 2-5 low-level task-specific clusters.

Look for:
- Framework-specific patterns (Django, Flask, pytest, etc.)
- Problem-specific patterns (imports, environment, migrations, etc.)
- Language-specific patterns (Python, JavaScript, etc.)

OUTPUT (JSON):
{{
  "parent_cluster": "{parent_cluster}",
  "low_level_clusters": [
    {{
      "cluster_name": "Django Stack Trace to Code",
      "segment_ids": [...],
      "framework": "Django",
      "description": "Following Django stack traces to locate bug...",
      "confidence": 0.92,
      "num_segments": 18
    }}
  ]
}}

Return ONLY the JSON, no other text."""

    def cluster_segments(
        self,
        all_segments: List[tuple[str, Dict[str, Any]]],  # (instance_id, segment)
        level: str,
        parent_cluster: Optional[str] = None,
    ) -> ClusteringResult:
        """
        Cluster segments at specified level using LLM

        Args:
            all_segments: List of (instance_id, segment_dict) tuples
            level: 'high', 'mid', or 'low'
            parent_cluster: Name of parent cluster (for mid/low levels)
        """
        print(f"\n{'=' * 70}")
        print(f"Clustering {len(all_segments)} segments at {level.upper()} level")
        if parent_cluster:
            print(f"Parent cluster: {parent_cluster}")
        print(f"{'=' * 70}")

        # Create summaries
        summaries = [
            self.create_segment_summary(seg, inst_id) for inst_id, seg in all_segments
        ]

        # Create prompt
        prompt = self.create_clustering_prompt(summaries, level, parent_cluster)

        print(f"Calling {self.model} for {level}-level clustering...")
        print(f"Prompt length: {len(prompt)} chars")

        # Call LLM
        raw_response = self.call_llm(prompt)

        print(f"Received response: {len(raw_response)} chars")

        # Parse response
        try:
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            json_str = raw_response[json_start:json_end]

            result_data = json.loads(json_str)

            # Extract clusters based on level
            if level == "high":
                clusters = result_data["high_level_clusters"]
            elif level == "mid":
                clusters = result_data["mid_level_clusters"]
            else:
                clusters = result_data["low_level_clusters"]

            print(f"✅ Created {len(clusters)} {level}-level clusters")

            # Print cluster summary
            print(f"\n{level.upper()}-Level Cluster Summary:")
            print("-" * 70)
            for cluster in clusters:
                print(f"  📦 {cluster['cluster_name']}")
                print(f"     Segments: {cluster['num_segments']}")
                print(f"     Confidence: {cluster['confidence']}")
                print(f"     Description: {cluster['description'][:80]}...")

            result = ClusteringResult(
                level=level,
                clusters=clusters,
                parent_cluster=parent_cluster,
                raw_response=raw_response,
            )

            return result

        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to parse LLM response: {e}")
            print(f"Response: {raw_response[:500]}...")
            raise

    def hierarchical_clustering(
        self, segmentation_results: List[SegmentationResult]
    ) -> Dict[str, Any]:
        """
        Stage 2: Perform hierarchical clustering (high → mid → low)

        Returns:
            Complete clustering hierarchy
        """
        print(f"\n{'=' * 70}")
        print("STAGE 2: HIERARCHICAL CLUSTERING")
        print(f"{'=' * 70}")

        # Prepare all segments
        all_segments = []
        segment_lookup = {}  # seg_id -> (instance_id, segment_dict)

        for result in segmentation_results:
            for seg in result.segments:
                seg_id = f"seg_{result.instance_id}_{seg['segment_id']}"
                all_segments.append((result.instance_id, seg))
                segment_lookup[seg_id] = (result.instance_id, seg)

        print(f"Total segments to cluster: {len(all_segments)}")

        # High-level clustering
        print(f"\n{'─' * 70}")
        print("STEP 1: High-level clustering")
        print(f"{'─' * 70}")
        high_level_result = self.cluster_segments(all_segments, level="high")

        # Save high-level clusters
        high_level_file = self.stage2_dir / "high_level_clusters.json"
        with open(high_level_file, "w") as f:
            json.dump(
                {
                    "level": "high",
                    "clusters": high_level_result.clusters,
                    "raw_response": high_level_result.raw_response,
                },
                f,
                indent=2,
            )

        # Mid-level clustering (for each high-level cluster)
        print(f"\n{'─' * 70}")
        print("STEP 2: Mid-level clustering")
        print(f"{'─' * 70}")

        mid_level_clusters = []
        for hl_cluster in high_level_result.clusters:
            print(f"\nProcessing high-level cluster: {hl_cluster['cluster_name']}")

            # Get segments for this cluster
            cluster_segments = [
                segment_lookup[seg_id]
                for seg_id in hl_cluster["segment_ids"]
                if seg_id in segment_lookup
            ]

            if len(cluster_segments) < 5:
                print(
                    f"  ⚠️  Too few segments ({len(cluster_segments)}), skipping mid-level clustering"
                )
                continue

            # Cluster at mid level
            ml_result = self.cluster_segments(
                cluster_segments, level="mid", parent_cluster=hl_cluster["cluster_name"]
            )

            for ml_cluster in ml_result.clusters:
                ml_cluster["parent_high_level"] = hl_cluster["cluster_name"]
                mid_level_clusters.append(ml_cluster)

        # Save mid-level clusters
        mid_level_file = self.stage2_dir / "mid_level_clusters.json"
        with open(mid_level_file, "w") as f:
            json.dump({"level": "mid", "clusters": mid_level_clusters}, f, indent=2)

        # Low-level clustering (for each mid-level cluster)
        print(f"\n{'─' * 70}")
        print("STEP 3: Low-level clustering")
        print(f"{'─' * 70}")

        low_level_clusters = []
        for ml_cluster in mid_level_clusters:
            print(f"\nProcessing mid-level cluster: {ml_cluster['cluster_name']}")

            # Get segments for this cluster
            cluster_segments = [
                segment_lookup[seg_id]
                for seg_id in ml_cluster["segment_ids"]
                if seg_id in segment_lookup
            ]

            if len(cluster_segments) < 3:
                print(
                    f"  ⚠️  Too few segments ({len(cluster_segments)}), skipping low-level clustering"
                )
                continue

            # Cluster at low level
            ll_result = self.cluster_segments(
                cluster_segments, level="low", parent_cluster=ml_cluster["cluster_name"]
            )

            for ll_cluster in ll_result.clusters:
                ll_cluster["parent_mid_level"] = ml_cluster["cluster_name"]
                ll_cluster["parent_high_level"] = ml_cluster["parent_high_level"]
                low_level_clusters.append(ll_cluster)

        # Save low-level clusters
        low_level_file = self.stage2_dir / "low_level_clusters.json"
        with open(low_level_file, "w") as f:
            json.dump({"level": "low", "clusters": low_level_clusters}, f, indent=2)

        # Create complete hierarchy
        hierarchy = {
            "high_level": high_level_result.clusters,
            "mid_level": mid_level_clusters,
            "low_level": low_level_clusters,
        }

        # Save complete hierarchy
        hierarchy_file = self.stage2_dir / "complete_hierarchy.json"
        with open(hierarchy_file, "w") as f:
            json.dump(hierarchy, f, indent=2)

        # Summary
        print(f"\n{'=' * 70}")
        print("STAGE 2 COMPLETE")
        print(f"{'=' * 70}")
        print(f"High-level clusters: {len(high_level_result.clusters)}")
        print(f"Mid-level clusters: {len(mid_level_clusters)}")
        print(f"Low-level clusters: {len(low_level_clusters)}")
        print(f"\n💾 Saved hierarchy to {hierarchy_file}")

        return hierarchy

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3: WORKFLOW EXTRACTION
    # ═══════════════════════════════════════════════════════════════════

    def get_full_segment_details(
        self,
        segment_ids: List[str],
        segmentation_results: List[SegmentationResult],
        trajectory_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Get full details for segments (including event details from trajectories)

        Args:
            segment_ids: List of segment IDs (e.g., "seg_instance1_2")
            segmentation_results: Results from Stage 1
            trajectory_dir: Directory with trajectory JSONL files
        """
        full_segments = []

        for seg_id in segment_ids:
            # Parse segment ID: seg_<instance_id>_<segment_num>
            parts = seg_id.split("_")
            instance_id = "_".join(parts[1:-1])
            segment_num = int(parts[-1])

            # Find the segment
            seg_result = next(
                (r for r in segmentation_results if r.instance_id == instance_id), None
            )
            if not seg_result:
                continue

            segment = next(
                (s for s in seg_result.segments if s["segment_id"] == segment_num), None
            )
            if not segment:
                continue

            # Load trajectory events
            traj_path = trajectory_dir / f"{instance_id}.jsonl"
            if not traj_path.exists():
                continue

            events = self.load_trajectory(traj_path)

            # Get events for this segment
            start = segment["start_event"]
            end = segment["end_event"]
            segment_events = events[start : end + 1]

            full_segments.append(
                {
                    "segment_id": seg_id,
                    "instance_id": instance_id,
                    "segment_metadata": segment,
                    "events": segment_events,
                }
            )

        return full_segments

    def format_segments_for_extraction(
        self, full_segments: List[Dict[str, Any]]
    ) -> str:
        """Format full segment details for workflow extraction"""
        lines = []

        for i, seg_data in enumerate(full_segments, 1):
            seg = seg_data["segment_metadata"]
            events = seg_data["events"]

            lines.append(f"{'━' * 70}")
            lines.append(f"SEGMENT {i} (from {seg_data['instance_id']}):")
            lines.append(f"Purpose: {seg['purpose']}")
            lines.append(f"Goal: {seg['goal']}")
            lines.append(f"Outcome: {seg['outcome']}")
            lines.append("")

            for j, event in enumerate(events):
                timestamp = event.get("timestamp", 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)

                action_type = event.get("action", "Unknown")
                reasoning = str(event.get("reasoning", event.get("thought", "")))[:300]
                observation = str(event.get("observation", ""))[:200]

                lines.append(f"Event {j} ({minutes:02d}:{seconds:02d}):")
                lines.append(f"  Action: {action_type}")
                if reasoning:
                    lines.append(f'  Reasoning: "{reasoning}"')
                if observation:
                    lines.append(f'  Observation: "{observation}"')
                lines.append("")

            lines.append(f"Segment outcome: {seg['outcome']}")
            lines.append("")

        return "\n".join(lines)

    def create_extraction_prompt(
        self, cluster_name: str, formatted_segments: str, cluster_info: Dict[str, Any]
    ) -> str:
        """Create prompt for workflow extraction"""
        return f"""You are extracting a reusable workflow from successful agent trajectories.

CLUSTER: "{cluster_name}"
LEVEL: Low (task-specific)
PARENT: {cluster_info.get("parent_mid_level", "Unknown")} → {cluster_info.get("parent_high_level", "Unknown")}
FRAMEWORK: {cluster_info.get("framework", "General")}

This cluster contains {cluster_info["num_segments"]} segments.

Here are detailed examples from this cluster:

{formatted_segments}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Extract a reusable workflow from these segments.

Analyze what's COMMON across successful attempts:
1. What triggers this workflow? (when should an agent use this?)
2. What are the steps? (in order)
3. What commands/actions are typically used?
4. What are the decision points? (if X happens, do Y)
5. How do you know it succeeded?
6. What can go wrong? (common pitfalls)

Think about:
- What worked consistently across instances?
- Are there variations that still lead to success?
- What are the critical steps vs. optional ones?

OUTPUT (JSON):
{{
  "workflow_name": "Descriptive Name",
  "workflow_id": "snake_case_id",
  "level": "low",
  "confidence": 0.92,
  "evidence": {{
    "num_segments": {cluster_info["num_segments"]},
    "appears_in_instances": <count unique instances>
  }},

  "trigger_conditions": {{
    "description": "When to use this workflow",
    "conditions": [
      "Condition 1",
      "Condition 2"
    ]
  }},

  "workflow_steps": [
    {{
      "step_number": 1,
      "action": "What to do",
      "description": "Detailed description",
      "example_commands": ["command1", "command2"],
      "typical_reasoning": "How agent thinks about this step",
      "is_critical": true
    }}
  ],

  "success_indicators": [
    "Indicator 1",
    "Indicator 2"
  ],

  "common_pitfalls": [
    {{
      "pitfall": "What can go wrong",
      "solution": "How to handle it"
    }}
  ],

  "metadata": {{
    "frameworks": ["Django"],
    "languages": ["Python"],
    "estimated_time": "30-60 seconds",
    "difficulty": "easy"
  }}
}}

Return ONLY the JSON, no other text."""

    def extract_workflow(
        self,
        cluster: Dict[str, Any],
        segmentation_results: List[SegmentationResult],
        trajectory_dir: Path,
        max_segments: int = 15,
    ) -> WorkflowResult:
        """
        Stage 3: Extract workflow from a cluster

        Args:
            cluster: Cluster dict with segment_ids
            segmentation_results: Results from Stage 1
            trajectory_dir: Directory with trajectories
            max_segments: Maximum number of segments to include
        """
        cluster_name = cluster["cluster_name"]
        segment_ids = cluster["segment_ids"]

        print(f"\n{'=' * 70}")
        print(f"Extracting workflow: {cluster_name}")
        print(f"{'=' * 70}")
        print(f"Total segments in cluster: {len(segment_ids)}")

        # Get full segment details
        full_segments = self.get_full_segment_details(
            segment_ids[:max_segments],  # Limit for token efficiency
            segmentation_results,
            trajectory_dir,
        )

        print(f"Retrieved {len(full_segments)} full segment details")

        if len(full_segments) < 2:
            print("⚠️  Too few segments with full details, skipping")
            return None

        # Format for LLM
        formatted = self.format_segments_for_extraction(full_segments)

        # Create prompt
        prompt = self.create_extraction_prompt(cluster_name, formatted, cluster)

        print(f"Calling {self.model} for workflow extraction...")
        print(f"Prompt length: {len(prompt)} chars")

        # Call LLM
        raw_response = self.call_llm(prompt)

        print(f"Received response: {len(raw_response)} chars")

        # Parse response
        try:
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            json_str = raw_response[json_start:json_end]

            workflow = json.loads(json_str)

            print(f"✅ Extracted workflow: {workflow['workflow_name']}")
            print(f"   Steps: {len(workflow['workflow_steps'])}")
            print(f"   Confidence: {workflow['confidence']}")

            result = WorkflowResult(
                workflow=workflow,
                cluster_name=cluster_name,
                num_segments=len(full_segments),
                raw_response=raw_response,
            )

            return result

        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Failed to parse workflow: {e}")
            print(f"Response: {raw_response[:500]}...")
            return None

    def extract_all_workflows(
        self,
        hierarchy: Dict[str, Any],
        segmentation_results: List[SegmentationResult],
        trajectory_dir: Path,
    ) -> List[WorkflowResult]:
        """
        Stage 3: Extract workflows from all low-level clusters
        """
        print(f"\n{'=' * 70}")
        print("STAGE 3: WORKFLOW EXTRACTION")
        print(f"{'=' * 70}")

        low_level_clusters = hierarchy["low_level"]
        print(f"Extracting workflows from {len(low_level_clusters)} low-level clusters")

        workflows = []
        for i, cluster in enumerate(low_level_clusters, 1):
            print(
                f"\n[{i}/{len(low_level_clusters)}] Processing cluster: {cluster['cluster_name']}"
            )

            # Skip clusters with too few segments
            if cluster["num_segments"] < 3:
                print(f"  ⚠️  Too few segments ({cluster['num_segments']}), skipping")
                continue

            try:
                result = self.extract_workflow(
                    cluster, segmentation_results, trajectory_dir
                )

                if result:
                    workflows.append(result)

                    # Save individual workflow
                    workflow_id = result.workflow["workflow_id"]
                    workflow_file = self.stage3_dir / f"{workflow_id}.json"
                    with open(workflow_file, "w") as f:
                        json.dump(
                            {
                                "workflow": result.workflow,
                                "cluster_name": result.cluster_name,
                                "num_segments": result.num_segments,
                                "raw_response": result.raw_response,
                            },
                            f,
                            indent=2,
                        )

            except Exception as e:
                print(f"❌ Error extracting workflow: {e}")
                continue

        # Save all workflows
        all_workflows_file = self.stage3_dir / "all_workflows.json"
        with open(all_workflows_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "num_workflows": len(workflows),
                    "workflows": [w.workflow for w in workflows],
                },
                f,
                indent=2,
            )

        # Summary
        print(f"\n{'=' * 70}")
        print("STAGE 3 COMPLETE")
        print(f"{'=' * 70}")
        print(f"Extracted {len(workflows)} workflows")
        print(f"\n💾 Saved all workflows to {all_workflows_file}")

        # Print workflow summary
        print("\nWorkflow Summary:")
        print("-" * 70)
        for w in workflows:
            wf = w.workflow
            print(f"  📋 {wf['workflow_name']}")
            print(f"     ID: {wf['workflow_id']}")
            print(f"     Steps: {len(wf['workflow_steps'])}")
            print(f"     Confidence: {wf['confidence']}")
            print(
                f"     Framework: {', '.join(wf['metadata'].get('frameworks', ['General']))}"
            )

        return workflows

    # ═══════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ═══════════════════════════════════════════════════════════════════

    def run_pipeline(self, trajectory_dir: Path, limit: Optional[int] = None):
        """
        Run complete 3-stage pipeline

        Args:
            trajectory_dir: Directory containing trajectory JSONL files
            limit: Optional limit on number of trajectories (for testing)
        """
        print(f"\n{'█' * 70}")
        print(f"{'█' * 70}")
        print(f"  LLM-FIRST WORKFLOW EXTRACTION PIPELINE (OpenAI {self.model})")
        print(f"{'█' * 70}")
        print(f"{'█' * 70}")
        print(f"\nTrajectory directory: {trajectory_dir}")
        print(f"Output directory: {self.output_dir}")
        if limit:
            print(f"Limit: {limit} trajectories")
        print()

        # Stage 1: Segmentation
        segmentation_results = self.segment_all_trajectories(
            trajectory_dir, limit=limit
        )

        # Stage 2: Clustering
        hierarchy = self.hierarchical_clustering(segmentation_results)

        # Stage 3: Workflow Extraction
        workflows = self.extract_all_workflows(
            hierarchy, segmentation_results, trajectory_dir
        )

        # Final summary
        print(f"\n{'█' * 70}")
        print("PIPELINE COMPLETE!")
        print(f"{'█' * 70}")
        print("📊 Final Statistics:")
        print(f"   Trajectories processed: {len(segmentation_results)}")
        print(
            f"   Total segments: {sum(len(r.segments) for r in segmentation_results)}"
        )
        print(f"   High-level clusters: {len(hierarchy['high_level'])}")
        print(f"   Mid-level clusters: {len(hierarchy['mid_level'])}")
        print(f"   Low-level clusters: {len(hierarchy['low_level'])}")
        print(f"   Workflows extracted: {len(workflows)}")
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"{'█' * 70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-First Workflow Extraction Pipeline (OpenAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on 3 trajectories
  python extractor_openai.py /path/to/trajectories --limit 3

  # Run on all trajectories
  python extractor_openai.py /path/to/trajectories

  # Use CMU AI Gateway
  python extractor_openai.py /path/to/trajectories \\
    --base-url https://ai-gateway.andrew.cmu.edu/ \\
    --model gpt-4 \\
    --limit 3

  # Use different model
  python extractor_openai.py /path/to/trajectories --model o1-mini

  # Specify API key directly
  python extractor_openai.py /path/to/trajectories \\
    --api-key "your-key" \\
    --model gpt-3.5-turbo
        """,
    )

    parser.add_argument(
        "trajectory_dir", type=Path, help="Directory containing trajectory JSONL files"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of trajectories to process (for testing)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./cawm_output_openai",
        help="Output directory (default: ./cawm_output_openai)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="o1",
        help="OpenAI model to use (default: o1, alternatives: o1-mini, gpt-4, gpt-3.5-turbo)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API base URL (e.g., https://ai-gateway.andrew.cmu.edu/)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    # Validate trajectory directory
    if not args.trajectory_dir.exists():
        print(f"❌ Error: Trajectory directory not found: {args.trajectory_dir}")
        return 1

    # Show configuration
    if args.base_url:
        print(f"Using custom API endpoint: {args.base_url}")

    # Create extractor
    extractor = WorkflowExtractor(
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output,
        base_url=args.base_url,
    )

    # Run pipeline
    try:
        extractor.run_pipeline(args.trajectory_dir, limit=args.limit)
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
