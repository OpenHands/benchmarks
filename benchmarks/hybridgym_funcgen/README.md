# Hybrid-Gym func_gen Evaluation

This module integrates the **func_gen** benchmark from [Hybrid-Gym](https://github.com/Hybrid-Gym/Hybrid-Gym). The agent is given a function with its body removed (replaced by a TODO stub) and must implement the function body based on the signature and docstring.

## Overview

Each instance provides:
- A Python repository (cloned at runtime from GitHub)
- A target function whose body has been replaced with `pass  # TODO: Implement this function`
- The function's signature and docstring remain intact

The agent must read the context, understand the specification from the docstring, and write a correct implementation. Evaluation uses RepoST's eval_script tests that compare the agent's implementation against the reference.

## Prerequisites

1. **Docker**: Required for workspace containers and evaluation (RepoST uses `yiqingxyq/repost:v0`).
2. **LLM API Key**: Configure your LLM provider credentials.

## Usage

### Running Inference

```bash
uv run hybridgym-funcgen-infer .llm_config/config.json
uv run hybridgym-funcgen-infer .llm_config/config.json --n-limit 5
uv run hybridgym-funcgen-infer .llm_config/config.json --workspace docker
```

### Evaluating Results

```bash
uv run hybridgym-funcgen-eval ./eval_outputs/.../output.jsonl --run-id my_run
uv run hybridgym-funcgen-eval ./eval_outputs/.../output.jsonl --run-id my_run --no-docker
```

## Dataset

- **HuggingFace**: [`hybrid-gym/hybrid_gym_func_gen`](https://huggingface.co/datasets/hybrid-gym/hybrid_gym_func_gen)
- **Default split**: `train`

## Evaluation Criteria

An instance is marked **resolved** when the generated implementation passes all tests in the RepoST eval_script. The eval_script renames the agent's function to `<name>_new_implementation` and compares its outputs against the reference implementation.

## References

- [Hybrid-Gym Paper](https://arxiv.org/abs/2602.16819v1) -- "Hybrid-Gym: Training Coding Agents to Generalize Across Tasks"
- [Hybrid-Gym GitHub](https://github.com/Hybrid-Gym/Hybrid-Gym)
