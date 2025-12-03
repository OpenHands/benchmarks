# CAWM - Code Agent Workflow Memory

A modular system for extracting **task-specific experiences** from agent execution trajectories.

## Overview

CAWM (Code Agent Workflow Memory) analyzes successful agent trajectories from benchmarks like SWE-Bench and extracts **actionable experiences** (not generic workflows) that can guide future agent behavior.

**Key Output**: Task-specific insights like:
- "When NDData arithmetic fails with mask propagation, check if mask is None before bitwise operations"
- "For HTTP date parsing, ensure RFC 7231 compliance - years 00-69 map to 2000-2069"

NOT generic advice like "First reproduce the bug" or "Write tests before fixing".

## Quick Start

### CLI Usage (Recommended)

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-..."

# Run with default settings
uv run python CAWM/main.py --output CAWM/workflow/my_run

# Run with custom settings
uv run python CAWM/main.py \
    --output CAWM/workflow/experiment1 \
    --clustering action_sequence \
    --threshold 0.7 \
    --limit 20 \
    --verbose
```

### Output Files

All files are created at pipeline start and updated progressively:

```
output_dir/
├── workflows.json           # Final extracted experiences
├── summary.txt              # Human-readable summary
├── clusters.json            # Clustering details (which trajectories grouped together)
├── induction_details.json   # Per-cluster induction results
└── pipeline_stats.json      # Timing, statistics, and status
```

Each file contains a `status` field: `pending` → `in_progress` → `completed` (or `failed`).

---

## CLI Reference

```bash
uv run python CAWM/main.py [OPTIONS]
```

### Required Options

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output directory for all files |

### Optional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | `CAWM/trajectories/resolved_trajectories.jsonl` | Input trajectory file |
| `--clustering`, `-k` | `problem_description` | Clustering method |
| `--threshold`, `-t` | `0.2` | Similarity threshold [0-1] |
| `--compression`, `-c` | `key_step_extraction` | Compression strategy |
| `--level`, `-l` | `general` | Workflow abstraction level |
| `--model`, `-m` | `moonshotai/kimi-k2-0905` | LLM model via OpenRouter |
| `--limit`, `-n` | None | Limit trajectories (for testing) |
| `--verbose`, `-v` | False | Enable debug logging |

---

## Parameter Guide: When to Use What

### Clustering Methods

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| `problem_description` | **Default choice**. Group by task semantics. | Semantic grouping, diverse experiences | Token-based, may miss nuance |
| `action_sequence` | Group by behavioral patterns (what the agent did). | Pattern-based, good with threshold 0.7 | N-gram similarity, can still cluster similar behaviors |
| `repository` | Group by source repo (django, astropy, etc.). | Clean separation by domain | Only useful if data spans multiple repos |
| `code_modification` | Group by modified files. | Component-focused experiences | Requires git_patch in data |
| `random` | Baseline testing only. | - | Not useful for production |

### Threshold Selection

The threshold controls how similar trajectories must be to cluster together:

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.2 - 0.3 | Few large clusters | Maximum diversity, fewer LLM calls |
| 0.4 - 0.5 | Medium clusters | Balanced |
| 0.6 - 0.7 | Many small clusters | Fine-grained patterns, more LLM calls |
| 0.8+ | Almost 1:1 | Each trajectory separate (not recommended) |

**Rule of thumb**:
- Want fewer LLM calls? Lower threshold (0.2-0.3)
- Want more diverse experiences? Higher threshold (0.6-0.7)

### Recommended Parameter Combinations

#### Combination 1: Quick Exploration (Default)
```bash
uv run python CAWM/main.py \
    --output output/quick \
    --clustering problem_description \
    --threshold 0.2 \
    --limit 10
```
- **Use when**: First run, understanding the data
- **Result**: 2-3 large clusters, ~10-15 experiences
- **LLM calls**: ~3

#### Combination 2: Diverse Experiences
```bash
uv run python CAWM/main.py \
    --output output/diverse \
    --clustering action_sequence \
    --threshold 0.7
```
- **Use when**: Want maximum diversity of experiences
- **Result**: Many small clusters (15-25), ~50-80 experiences
- **LLM calls**: ~20-25

#### Combination 3: Domain-Specific (Multi-Repo Data)
```bash
uv run python CAWM/main.py \
    --output output/by_repo \
    --clustering repository
```
- **Use when**: Data spans multiple repositories
- **Result**: One cluster per repo
- **LLM calls**: Number of unique repos

#### Combination 4: File-Based Grouping
```bash
uv run python CAWM/main.py \
    --output output/by_files \
    --clustering code_modification \
    --threshold 0.3
```
- **Use when**: Want to group by affected components
- **Result**: Clusters of trajectories that modified similar files
- **LLM calls**: Varies

---

## Core Modules

### 1. ClusteringModule

Groups similar trajectories. Uses **n-gram Jaccard similarity** (not simple set comparison) for `action_sequence` to preserve sequential patterns.

```python
from CAWM import ClusteringModule, SimilarityMethod

# Action sequence clustering (uses bigram + trigram patterns)
clusterer = ClusteringModule(method=SimilarityMethod.ACTION_SEQUENCE)
clusterer.config.threshold = 0.7

# Problem description clustering (token-based Jaccard)
clusterer = ClusteringModule(method=SimilarityMethod.PROBLEM_DESCRIPTION)
clusterer.config.threshold = 0.2

# Repository-based (groups by repo name from instance_id)
clusterer = ClusteringModule(method=SimilarityMethod.REPOSITORY)

clusters = clusterer.cluster(trajectories)
```

### 2. CompressionModule

Reduces trajectory length while preserving key information.

| Strategy | LLM Required | Description |
|----------|--------------|-------------|
| `KEY_STEP_EXTRACTION` | No | Keeps file edits, tests, and context |
| `ACTION_TYPE_FILTERING` | No | Keeps only specified action types |
| `HIERARCHICAL_SUMMARIZATION` | Yes | LLM-based chunk summarization |
| `NO_OP` | No | Pass-through (no compression) |

```python
from CAWM import CompressionModule, CompressionStrategy

# Fast, no LLM needed
compressor = CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)
compressed = compressor.compress_batch(trajectories)
```

### 3. InductionModule

Uses LLM to extract experiences from trajectory clusters.

```python
from CAWM import InductionModule, WorkflowLevel

inductor = InductionModule(llm_client=llm_client)

# Extract from clusters
workflows = inductor.induce_from_clusters(clusters, level=WorkflowLevel.GENERAL)
```

**Output Format** (experiences, not generic workflows):
```json
{
  "experiences": [
    {
      "trigger": "When NDData arithmetic fails with mask propagation...",
      "insight": "The mask propagation logic assumes both operands have masks...",
      "action": "Check if operand.mask exists before accessing properties...",
      "category": "debugging"
    }
  ]
}
```

### 4. CAWMPipeline

Orchestrates the full process with standardized output.

```python
from CAWM import CAWMPipeline, PipelineConfig, LLMClient

llm_client = LLMClient(provider="openrouter", model="moonshotai/kimi-k2-0905")

config = PipelineConfig(
    compression_strategy=CompressionStrategy.KEY_STEP_EXTRACTION,
    clustering_method=SimilarityMethod.ACTION_SEQUENCE,
    clustering_threshold=0.7,
    workflow_level=WorkflowLevel.GENERAL,
)

pipeline = CAWMPipeline(
    llm_client=llm_client,
    config=config,
    output_dir="output/my_run"  # All files saved here
)

workflows = pipeline.run(trajectories)
```

---

## Output File Details

### `clusters.json`

Shows which trajectories were grouped together:

```json
{
  "metadata": {
    "status": "completed",
    "clustering_method": "action_sequence",
    "threshold": 0.7,
    "num_clusters": 20
  },
  "clusters": [
    {
      "cluster_id": "seq_cluster_0",
      "size": 3,
      "trajectories": [
        {
          "instance_id": "django__django-12345",
          "repository": "django",
          "instruction_preview": "Fix the bug in...",
          "action_types": ["FILE_EDIT", "TESTING", "EXPLORATION"]
        }
      ]
    }
  ]
}
```

### `induction_details.json`

Shows what experiences came from each cluster:

```json
{
  "metadata": {
    "status": "completed",
    "num_clusters_processed": 20,
    "total_experiences": 75
  },
  "cluster_results": [
    {
      "cluster_id": "seq_cluster_0",
      "trajectory_ids": ["django__django-12345", "django__django-12346"],
      "num_experiences_extracted": 4,
      "experiences": [
        {
          "trigger": "When...",
          "insight": "The root cause is...",
          "action": "Fix by..."
        }
      ],
      "duration_seconds": 2.5
    }
  ]
}
```

### `pipeline_stats.json`

Overall statistics and timing:

```json
{
  "status": "completed",
  "total_duration_seconds": 45.2,
  "stages": {
    "input": {"num_trajectories": 41},
    "clustering": {
      "num_clusters": 20,
      "cluster_sizes": [3, 2, 2, 1, ...],
      "duration_seconds": 0.01
    },
    "induction": {
      "num_experiences": 75,
      "experiences_per_cluster": [4, 3, 5, ...],
      "duration_seconds": 45.1
    }
  }
}
```

---

## Troubleshooting

### "API key not found"
```bash
export OPENROUTER_API_KEY="sk-or-..."
```
Or prefix command: `OPENROUTER_API_KEY="..." uv run python ...`

### All trajectories in one cluster
- **Cause**: Threshold too low or clustering method not suitable
- **Fix**: Increase threshold (0.5 → 0.7) or try `action_sequence` clustering

### Too many small clusters (1-2 trajectories each)
- **Cause**: Threshold too high
- **Fix**: Decrease threshold (0.7 → 0.4)

### Generic experiences ("write tests first", "reproduce the bug")
- **Cause**: Induction prompt issue (already fixed in current version)
- **Verify**: Check `induction_details.json` for experience quality

---

## Architecture

```
Trajectories (JSONL)
        │
        ▼
┌──────────────────┐
│  CompressionModule │  ← Reduce trajectory length
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  ClusteringModule  │  ← Group similar trajectories
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  InductionModule   │  ← Extract experiences via LLM
└──────────────────┘
        │
        ▼
   Output Files
   (workflows.json, clusters.json, etc.)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for LLM access |

---

## Tips

1. **Start with `--limit 5`**: Test with small samples before full runs
2. **Check `clusters.json` first**: Understand how trajectories are grouped
3. **Lower threshold = fewer LLM calls**: Good for cost optimization
4. **Use `action_sequence` with threshold 0.7**: Best diversity for SWE-bench data
5. **Check `pipeline_stats.json`**: See timing and cluster distribution
