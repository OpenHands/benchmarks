# Hybrid-Gym issue_localize Evaluation

This module integrates the **issue_localize** benchmark from [Hybrid-Gym](https://github.com/Hybrid-Gym/Hybrid-Gym). Given a GitHub issue description, the agent must locate relevant code in the repository and add comments explaining why each location is related to the issue.

## Overview

Each instance provides:
- A Python repository (cloned at runtime from GitHub)
- A GitHub issue description (problem statement)
- A gold patch indicating which files need modification

The agent must explore the codebase, identify files related to the issue, and add comments at relevant locations. Success requires touching at least one file from the gold patch while making only comment changes.

## Usage

### Running Inference

```bash
uv run hybridgym-issuelocalize-infer .llm_config/config.json
uv run hybridgym-issuelocalize-infer .llm_config/config.json --n-limit 5
uv run hybridgym-issuelocalize-infer .llm_config/config.json --workspace docker
```

### Evaluating Results

```bash
uv run hybridgym-issuelocalize-eval ./eval_outputs/.../output.jsonl --run-id my_run
```

## Dataset

- **HuggingFace**: [`SWE-Gym/SWE-Gym-Raw`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Raw)
- **Default split**: `train`

## Evaluation Criteria

An instance is marked **resolved** when both conditions are met:

| Criterion | Description |
|-----------|-------------|
| `localization` | At least one file from the gold patch was touched by the agent |
| `comments_only` | All changes are comments only (no code modifications); inline comments appended to existing lines are allowed |

## References

- [Hybrid-Gym Paper](https://arxiv.org/abs/2602.16819v1) -- "Hybrid-Gym: Training Coding Agents to Generalize Across Tasks"
- [Hybrid-Gym GitHub](https://github.com/Hybrid-Gym/Hybrid-Gym)
