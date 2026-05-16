# Hybrid-Gym dep_search Evaluation

This module integrates the **dep_search** benchmark from [Hybrid-Gym](https://github.com/Hybrid-Gym/Hybrid-Gym). The agent must analyze a target function, identify all functions and classes it directly calls within the repository, and annotate each dependency with a comment above its definition.

## Overview

Each instance provides:
- A Python repository (cloned at runtime from GitHub)
- A target function name, file path, and line number

The agent must read the target function, trace its calls, and add a comment (`# this function/class is called by the <name> function`) above each called module's definition. Success is measured by precision, recall, and F1 across the set of expected dependencies.

## Usage

### Running Inference

```bash
uv run hybridgym-depsearch-infer .llm_config/config.json
uv run hybridgym-depsearch-infer .llm_config/config.json --n-limit 5
uv run hybridgym-depsearch-infer .llm_config/config.json --workspace docker
```

### Evaluating Results

```bash
uv run hybridgym-depsearch-eval ./eval_outputs/.../output.jsonl --run-id my_run
```

## Dataset

- **HuggingFace**: [`hybrid-gym/hybrid_gym_dep_search`](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_dep_search)
- **Default split**: `train`

## Evaluation Criteria

An instance is marked **fully resolved** when all conditions are met:

| Criterion | Description |
|-----------|-------------|
| All dependencies annotated | Every expected dependency has a correctly placed comment (recall = 1.0) |
| No false positives | No comments placed for non-dependencies |
| No duplicates | Each dependency annotated exactly once |
| Comments only | All changes are comments (no code modifications) |

Partial credit is given via precision/recall/F1 scores.

## References

- [Hybrid-Gym Paper](https://arxiv.org/abs/2602.16819v1) -- "Hybrid-Gym: Training Coding Agents to Generalize Across Tasks"
- [Hybrid-Gym GitHub](https://github.com/Hybrid-Gym/Hybrid-Gym)
