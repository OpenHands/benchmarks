# CAWM - Code Agent Workflow Memory

A modular system for extracting reusable workflows from agent execution trajectories.

## Overview

CAWM (Code Agent Workflow Memory) analyzes successful agent trajectories from benchmarks like SWE-Bench and extracts generalizable workflows that can guide future agent behavior. The system consists of three core modules:

1. **CompressionModule** - Reduces trajectory length while preserving key information
2. **ClusteringModule** - Groups similar trajectories for pattern discovery
3. **InductionModule** - Uses LLM to induce reusable workflows from clusters

## Quick Start

### Prerequisites

```bash
# Set your OpenRouter API key (required for LLM-based features)
export OPENROUTER_API_KEY="sk-or-..." 
```

### Basic Usage

```python
from CAWM import (
    CAWMPipeline,
    PipelineConfig,
    LLMClient,
    Trajectory,
    CompressionStrategy,
    SimilarityMethod,
    WorkflowLevel
)

# 1. Initialize LLM Client
llm_client = LLMClient(provider="openrouter", model="moonshotai/kimi-k2-0905") # Using Kimi via OpenRouter

# 2. Configure Pipeline
config = PipelineConfig(
    compression_strategy=CompressionStrategy.KEY_STEP_EXTRACTION,
    clustering_method=SimilarityMethod.ACTION_SEQUENCE,
    workflow_level=WorkflowLevel.GENERAL,
    clustering_threshold=0.5
)

# 3. Create Pipeline
pipeline = CAWMPipeline(llm_client=llm_client, config=config)

# 4. Run on Data
workflows = pipeline.run_from_file(
    input_path="CAWM/trajectories/resolved_trajectories.jsonl",
    output_path="CAWM/workflow/extracted_workflows.json"
)

print(f"Extracted {len(workflows)} workflows")
```

### Run Demo Script

```bash
# Run the included demo
uv run python tests/run_cawm_demo.py
```

---

## Module Reference

### LLMClient

Unified client supporting OpenRouter API with automatic retry and error handling.

```python
from CAWM import LLMClient

# OpenRouter (default, recommended)
client = LLMClient(
    provider="openrouter",
    model="moonshotai/kimi-k2-0905",  # Kimi model
    temperature=0.0,
    max_tokens=4096,
    timeout=60.0,      # Request timeout in seconds
    max_retries=3      # Automatic retry with exponential backoff
)
```

**Features:**
- Automatic API key detection from environment variables
- Exponential backoff retry for rate limits and server errors
- Handles standard API errors

---

### CompressionModule

Reduces trajectory length using one of three strategies. Strategies can be composed.

```python
from CAWM import CompressionModule, CompressionStrategy, CompressionConfig

# Strategy 1: Key Step Extraction (default, no LLM needed)
# Keeps only file edits, tests, and 1 step of context around them
compressor = CompressionModule(
    strategy=CompressionStrategy.KEY_STEP_EXTRACTION
)

# Strategy 2: Action Type Filtering (no LLM needed)
# Keeps only events of specified action types
config = CompressionConfig(
    keep_action_types=[ActionType.FILE_EDIT, ActionType.TESTING]
)
compressor = CompressionModule(
    strategy=CompressionStrategy.ACTION_TYPE_FILTERING,
    config=config
)

# Strategy 3: Hierarchical Summarization (requires LLM)
# Chunks trajectory and summarizes each chunk via LLM
config = CompressionConfig(
    chunk_size=10,  # Events per chunk
    summary_prompt_template="Summarize these actions into high-level steps:"
)
compressor = CompressionModule(
    strategy=CompressionStrategy.HIERARCHICAL_SUMMARIZATION,
    llm_client=llm_client,
    config=config
)

# Compress a trajectory
compressed = compressor.compress(trajectory)

# Compress batch
compressed_batch = compressor.compress_batch(trajectories)
```

**Strategy Composition:**

```python
# Compose multiple strategies (executed in sequence)
step1 = CompressionModule(strategy=CompressionStrategy.ACTION_TYPE_FILTERING)
step2 = CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)

composed = step1 + step2  # First filter, then extract key steps
result = composed.compress(trajectory)
```

| Strategy | LLM Required | Use Case |
|----------|--------------|----------|
| `KEY_STEP_EXTRACTION` | No | Fast, preserves edit/test context |
| `ACTION_TYPE_FILTERING` | No | Remove noise (navigation, setup) |
| `HIERARCHICAL_SUMMARIZATION` | Yes | Maximum compression, semantic preservation |
| `NO_OP` | No | Pass-through (no compression) |

---

### ClusteringModule

Groups similar trajectories for pattern discovery.

```python
from CAWM import ClusteringModule, SimilarityMethod, ClusteringConfig

# Method 1: Action Sequence Similarity (default)
# Jaccard similarity of action types
clusterer = ClusteringModule(
    method=SimilarityMethod.ACTION_SEQUENCE
)
clusterer.config.threshold = 0.5  # Similarity threshold [0, 1]

# Method 2: Problem Description Similarity
# Jaccard similarity of tokenized instruction text
clusterer = ClusteringModule(
    method=SimilarityMethod.PROBLEM_DESCRIPTION
)

# Method 3: Code Modification Similarity
# Jaccard similarity of modified file paths (from git_patch)
clusterer = ClusteringModule(
    method=SimilarityMethod.CODE_MODIFICATION
)

# Method 4: Random (for baseline/testing)
clusterer = ClusteringModule(
    method=SimilarityMethod.RANDOM
)

# Cluster trajectories
clusters = clusterer.cluster(trajectories)

# Get pairwise similarity
sim = clusterer.get_similarity(traj1, traj2)
```

| Method | Use Case |
|--------|----------|
| `ACTION_SEQUENCE` | Group by behavioral pattern |
| `PROBLEM_DESCRIPTION` | Group by task semantics |
| `CODE_MODIFICATION` | Group by affected files |
| `RANDOM` | Baseline/testing |

---

### InductionModule

Uses LLM to induce reusable workflows from trajectory clusters.

```python
from CAWM import InductionModule, InductionConfig, WorkflowLevel

# Configure induction
config = InductionConfig(
    level=WorkflowLevel.GENERAL,
    max_workflows=5,   # Max workflows per cluster
    min_steps=2,       # Minimum steps per workflow
    max_steps=10       # Maximum steps per workflow
)

inductor = InductionModule(
    llm_client=llm_client,
    config=config
)

# Induce from trajectories
workflows = inductor.induce(trajectories, level=WorkflowLevel.GENERAL)

# Induce from clusters
workflows = inductor.induce_from_clusters(clusters, level=WorkflowLevel.SPECIFIC)

# Induce both levels at once
result = inductor.induce_hierarchical(trajectories)
# result = {WorkflowLevel.GENERAL: [...], WorkflowLevel.SPECIFIC: [...]}
```

**Workflow Levels:**

| Level | Description | Use Case |
|-------|-------------|----------|
| `GENERAL` | Cross-project, highly abstracted (placeholders like `{file}`, `{func}`) | Reusable across any codebase |
| `SPECIFIC` | Retains project/issue context, more detailed patterns | Project-specific guidance |

---

### CAWMPipeline

Orchestrates the full workflow: Load -> Compress -> Cluster -> Induce -> Save

```python
from CAWM import CAWMPipeline, PipelineConfig

# Full configuration
config = PipelineConfig(
    # Compression
    compression_strategy=CompressionStrategy.KEY_STEP_EXTRACTION,

    # Clustering
    clustering_method=SimilarityMethod.ACTION_SEQUENCE,
    clustering_threshold=0.5,

    # Induction
    workflow_level=WorkflowLevel.GENERAL,

    # LLM (used if client not passed explicitly)
    llm_model="moonshotai/kimi-k2-0905"
)

pipeline = CAWMPipeline(llm_client=llm_client, config=config)

# Option 1: Run from file
workflows = pipeline.run_from_file(
    input_path="CAWM/trajectories/resolved_trajectories.jsonl",
    output_path="output/workflows.json"  # Optional
)

# Option 2: Run from loaded trajectories
trajectories = Trajectory.load_from_jsonl("path/to/data.jsonl")
workflows = pipeline.run(trajectories)
```

---

## Configuration Reference

### PipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression_strategy` | `CompressionStrategy` | `KEY_STEP_EXTRACTION` | Compression method |
| `clustering_method` | `SimilarityMethod` | `ACTION_SEQUENCE` | Clustering similarity |
| `clustering_threshold` | `float` | `0.5` | Similarity threshold [0, 1] |
| `workflow_level` | `WorkflowLevel` | `GENERAL` | Abstraction level |
| `llm_model` | `str` | `"moonshotai/kimi-k2-0905"` | Default model |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API access |

---

## Common Usage Patterns

### Pattern 1: Fast Local Processing (No LLM)

```python
from CAWM import (
    CompressionModule, ClusteringModule,
    CompressionStrategy, SimilarityMethod, Trajectory
)

# Load data
trajectories = Trajectory.load_from_jsonl("data.jsonl")

# Compress without LLM
compressor = CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)
compressed = compressor.compress_batch(trajectories)

# Cluster without LLM
clusterer = ClusteringModule(method=SimilarityMethod.ACTION_SEQUENCE)
clusters = clusterer.cluster(compressed)

print(f"Formed {len(clusters)} clusters from {len(trajectories)} trajectories")
for c in clusters:
    print(f"  {c.cluster_id}: {len(c)} trajectories")
```

### Pattern 2: Full LLM Pipeline

```python
from CAWM import CAWMPipeline, PipelineConfig, LLMClient, CompressionStrategy

llm = LLMClient(provider="openrouter", model="moonshotai/kimi-k2-0905")
config = PipelineConfig(
    compression_strategy=CompressionStrategy.HIERARCHICAL_SUMMARIZATION,
    clustering_threshold=0.3
)

pipeline = CAWMPipeline(llm_client=llm, config=config)
workflows = pipeline.run_from_file("data.jsonl", "output.json")
```

### Pattern 3: Custom Module Composition

```python
from CAWM import (
    CompressionModule, ClusteringModule, InductionModule,
    CompressionStrategy, SimilarityMethod, WorkflowLevel,
    LLMClient, Trajectory
)

llm = LLMClient(provider="openrouter", model="moonshotai/kimi-k2-0905")
trajectories = Trajectory.load_from_jsonl("data.jsonl")

# Step 1: Custom compression chain
compress1 = CompressionModule(strategy=CompressionStrategy.ACTION_TYPE_FILTERING)
compress2 = CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)
composed_compressor = compress1 + compress2

compressed = composed_compressor.compress_batch(trajectories)

# Step 2: Cluster by code modifications
clusterer = ClusteringModule(method=SimilarityMethod.CODE_MODIFICATION)
clusterer.config.threshold = 0.2  # Loose threshold
clusters = clusterer.cluster(compressed)

# Step 3: Induce specific workflows
inductor = InductionModule(llm_client=llm)
workflows = inductor.induce_from_clusters(clusters, level=WorkflowLevel.SPECIFIC)
```

### Pattern 4: Analyze Trajectory Similarity

```python
from CAWM import ClusteringModule, SimilarityMethod, Trajectory

# Load two trajectories to compare
t1 = Trajectory.load_from_jsonl("file1.jsonl")[0]
t2 = Trajectory.load_from_jsonl("file2.jsonl")[0]

# Compare by different metrics
for method in SimilarityMethod:
    if method == SimilarityMethod.RANDOM:
        continue
    clusterer = ClusteringModule(method=method)
    sim = clusterer.get_similarity(t1, t2)
    print(f"{method.value}: {sim:.3f}")
```

---

## Output Format

Workflows are saved in JSON format compatible with existing AWM integration:

```json
{
  "workflows": [
    {
      "id": "wf-general-abc123",
      "description": "Explore codebase to understand structure",
      "category": "exploration",
      "level": 1,
      "steps": [
        {
          "env_description": "Initial state, unknown codebase",
          "reasoning": "Need to understand project structure",
          "action": "find {repo} -type f -name '*.py'",
          "action_type": "exploration"
        },
        {
          "env_description": "Found Python files",
          "reasoning": "Read main entry point",
          "action": "view {file}",
          "action_type": "file_view"
        }
      ],
      "source_instances": ["repo__issue-1", "repo__issue-2"],
      "frequency": 1,
      "pattern": ["exploration", "file_view"]
    }
  ],
  "count": 1,
  "config": {
    "level": "GENERAL",
    "clustering": "ACTION_SEQUENCE"
  }
}
```

---

## Data Format

### Input: Trajectory JSONL

Each line is a JSON object with:

```json
{
  "instance_id": "django__django-12345",
  "instruction": "Fix the bug in models.py...",
  "history": [
    {
      "kind": "ActionEvent",
      "action": {"kind": "TerminalAction", "command": "ls -la"},
      "thought": [{"text": "List files to explore"}]
    },
    {
      "kind": "ActionEvent",
      "action": {"kind": "FileEditorAction", "command": "view", "path": "models.py"},
      "thought": [{"text": "View the model file"}]
    }
  ],
  "test_result": {
    "git_patch": "diff --git a/models.py..."
  }
}
```

---

## Tips

1. **Start with defaults**: `PipelineConfig()` provides sensible defaults
2. **Adjust threshold**: Lower `clustering_threshold` = fewer, larger clusters
3. **Check trajectory count**: LLM costs scale with trajectories; test with small samples first
4. **Use KEY_STEP_EXTRACTION**: Fastest compression, good quality
5. **PROBLEM_DESCRIPTION clustering**: Best for grouping semantically similar tasks
6. **CODE_MODIFICATION clustering**: Best for grouping by affected components
