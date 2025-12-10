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

Reduces trajectory length while preserving key information. This is important because raw trajectories can have 50-200+ events, which would exceed LLM context limits.

#### Compression Strategies

| Strategy | LLM Required | Compression Ratio | Best For |
|----------|--------------|-------------------|----------|
| `KEY_STEP_EXTRACTION` | No | ~70-80% reduction | Default, fast processing |
| `ACTION_TYPE_FILTERING` | No | ~60-90% reduction | Focused analysis |
| `HIERARCHICAL_SUMMARIZATION` | Yes | ~80-90% reduction | Semantic preservation |
| `NO_OP` | No | 0% (pass-through) | Debugging, small trajectories |

#### Strategy Details

**1. KEY_STEP_EXTRACTION** (Default, Recommended)

Core logic:
1. Identify "key steps" - events that are FILE_EDIT or TESTING actions
2. For each key step, keep 1 event before and 1 after for context
3. If no key steps found, keep only first and last event

```
Original: [E1, E2, E3, EDIT, E5, E6, TEST, E8, E9, E10]
                   ↓
Compressed: [E3, EDIT, E5, E6, TEST, E8]
            (context before/after key steps)
```

Pros:
- No LLM calls, very fast
- Preserves the most important actions (edits and tests)
- Maintains context around key decisions

**2. ACTION_TYPE_FILTERING**

Core logic:
1. Filter events by action type whitelist
2. Default whitelist: `FILE_EDIT`, `TESTING`, `EXPLORATION`
3. All other event types are discarded

```
Original: [THINK, EXPLORE, EDIT, THINK, TEST, NAVIGATE, EDIT]
                       ↓
Compressed: [EXPLORE, EDIT, TEST, EDIT]
            (only whitelisted types)
```

Pros:
- Very aggressive compression
- Customizable via `keep_action_types` config

Cons:
- May lose important context (thoughts, navigation reasoning)

**3. HIERARCHICAL_SUMMARIZATION** (Requires LLM)

Core logic:
1. Split events into chunks (default: 10 events per chunk)
2. Send each chunk to LLM for summarization
3. Replace chunk with a single "summary event"

```
Original: [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, ...]
                              ↓
Compressed: [Summary("Steps 1-10: Explored codebase, found bug in X"),
             Summary("Steps 11-20: Fixed bug, ran tests")]
```

Pros:
- Preserves semantic meaning
- Best for very long trajectories

Cons:
- Requires LLM calls (slower, costs tokens)
- Summary quality depends on LLM

**4. NO_OP**

Pass-through, no compression. Use for debugging or when trajectories are already small.

#### Usage Example

```python
from CAWM import CompressionModule, CompressionStrategy

# Default: key step extraction (fast, no LLM)
compressor = CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)

# Custom filtering
from CAWM.models import ActionType
compressor = CompressionModule(strategy=CompressionStrategy.ACTION_TYPE_FILTERING)
compressor.config.keep_action_types = [ActionType.FILE_EDIT, ActionType.TESTING]

# Composition: chain multiple strategies
comp1 = CompressionModule(strategy=CompressionStrategy.ACTION_TYPE_FILTERING)
comp2 = CompressionModule(strategy=CompressionStrategy.KEY_STEP_EXTRACTION)
composed = comp1 + comp2  # Apply filtering first, then key extraction

compressed = compressor.compress_batch(trajectories)
```

---

### 3. InductionModule

Uses LLM to extract **task-specific experiences** from trajectory clusters. This is the core value-extraction step.

#### How It Works

```
Input: Cluster of similar trajectories
           ↓
Step 1: Compress trajectories (via CompressionModule)
           ↓
Step 2: Build prompt with trajectory context
           ↓
Step 3: LLM extracts experiences
           ↓
Step 4: Parse JSON response into Workflow objects
           ↓
Output: List of actionable experiences
```

#### Abstraction Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `GENERAL` | Cross-project experiences with placeholders like `{file}`, `{function}` | Default, reusable across projects |
| `SPECIFIC` | Detailed experiences with exact module/function names | Project-specific knowledge bases |

#### The Prompt Strategy

The induction prompt is designed to extract **specific, actionable insights** rather than generic advice:

**Good experiences** (what we want):
- "When encountering 'FieldError: Cannot resolve keyword', check if the field name conflicts with a reverse relation"
- "For HTTP date parsing, years 00-69 map to 2000-2069 per RFC 7231"
- "When inner class serialization fails in migrations, use `__qualname__` instead of `__name__`"

**Bad experiences** (filtered out):
- "First reproduce the bug" (too obvious)
- "Write tests before fixing" (generic advice)
- "Read the error message carefully" (not actionable)

#### Processing Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max trajectories per cluster | 10 | Prevents context overflow |
| Max events per trajectory | 20 | Focuses on key actions |
| Experiences per cluster | 3-5 | Balances quality vs quantity |

#### Output Format

```json
{
  "experiences": [
    {
      "trigger": "When/If <specific situation or error>",
      "insight": "The key discovery or root cause",
      "action": "Specific recommended approach or fix",
      "category": "debugging|refactoring|testing|configuration|api_usage"
    }
  ]
}
```

#### Categories

| Category | Description |
|----------|-------------|
| `debugging` | Finding and diagnosing issues |
| `refactoring` | Code structure improvements |
| `testing` | Test-related insights |
| `configuration` | Setup and config issues |
| `api_usage` | Correct API/library usage |

#### Usage Example

```python
from CAWM import InductionModule, WorkflowLevel

inductor = InductionModule(llm_client=llm_client)

# Extract general experiences (cross-project)
workflows = inductor.induce_from_clusters(clusters, level=WorkflowLevel.GENERAL)

# Extract specific experiences (project-focused)
workflows = inductor.induce_from_clusters(clusters, level=WorkflowLevel.SPECIFIC)

# Extract both levels at once
hierarchical = inductor.induce_hierarchical(trajectories)
# Returns: {WorkflowLevel.GENERAL: [...], WorkflowLevel.SPECIFIC: [...]}
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
