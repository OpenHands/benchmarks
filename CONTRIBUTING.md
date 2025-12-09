# Contributing to OpenHands Benchmarks

Thank you for your interest in contributing to OpenHands Benchmarks! This document provides guidelines for contributing to the project.

## Naming Conventions

To maintain consistency across the codebase, we follow these naming conventions:

### Benchmark Names

- **Code and file structure**: Use lowercase names without underscores or dashes
  - Directory names: `swebench`, `swtbench`, `commit0`
  - Python module names: `benchmarks.swebench`, `benchmarks.swtbench`, `benchmarks.commit0`
  - CLI commands: `swebench-infer`, `swtbench-eval`, `commit0-infer`

- **Documentation and README purposes**: Use the full name with dashes only
  - Documentation titles: "SWE-Bench", "SWT-Bench", "Commit-0"
  - README references: "SWE-Bench evaluation", "SWT-Bench benchmark"

### Examples

✅ **Correct usage:**
```python
# In code
from benchmarks.swebench.run_infer import main
from benchmarks.swtbench.eval_infer import evaluate

# In CLI
uv run swebench-infer config.json
uv run swtbench-eval output.jsonl

# In documentation
# SWE-Bench Benchmark Evaluation
This directory contains the SWE-Bench evaluation implementation.
```

❌ **Avoid:**
```python
# Don't mix naming conventions
from benchmarks.swe_bench.run_infer import main  # Use swebench instead
from benchmarks.swt_bench.eval_infer import evaluate  # Use swtbench instead

# Don't use underscores in CLI commands
uv run swe_bench-infer config.json  # Use swebench-infer instead
```

### Rationale

This convention eliminates the need for name normalization code and reduces complexity in the codebase. It ensures:
- Consistent import paths
- Simplified CLI command names
- Clear separation between code identifiers and human-readable documentation

## Development Setup

Before contributing, please set up your development environment:

```bash
make build
```

This will initialize the Agent SDK submodule and install all dependencies.

## Code Quality

- Follow the existing code style and formatting
- Run pre-commit hooks before submitting changes
- Write clear, focused commit messages
- Add tests for new functionality where appropriate

## Pull Requests

When submitting a pull request:
1. Follow the naming conventions outlined above
2. Update documentation if you're adding new features
3. Ensure all tests pass
4. Provide a clear description of your changes

Thank you for helping improve OpenHands Benchmarks!