# Contributing to OpenHands Benchmarks

This guide covers conventions and patterns for adding new benchmarks or modifying existing ones.

## Directory Structure

Each benchmark lives in its own folder under `benchmarks/`:

```
benchmarks/<benchmark_name>/
├── README.md              # Benchmark documentation
├── __init__.py
├── config.py              # Default configurations (INFER_DEFAULTS, EVAL_DEFAULTS)
├── run_infer.py           # Inference entrypoint
├── eval_infer.py          # Evaluation entrypoint
├── build_images.py        # Docker image building (if needed)
└── prompts/               # Prompt templates
```

**One benchmark per folder.** For similar benchmarks (e.g., SWE-bench and SWE-bench MultiModal), it's preferable to duplicate code than to merge them.
**benchmark_name should be lower case slug** No dashes, nor underscores, so SWE-bench MultiModal becomes swebenchmultimodal.

## Required Files

### run_infer.py

- Implements an `Evaluation` subclass with:
  - `prepare_instances()` → returns list of `EvalInstance`
  - `prepare_workspace(instance)` → returns `RemoteWorkspace`
  - `evaluate_instance(instance, workspace)` → returns `EvalOutput`
- Has a `main()` function as entrypoint
- Uses `get_parser()` from `benchmarks.utils.args_parser`
- Uses `EvalMetadata` model for configuration
- Handles both `docker` and `remote` workspace types

### eval_infer.py

- Converts inference output to the benchmark's evaluation format
- Runs the evaluation harness
- Generates a cost report

### build_images.py (when applicable)

- Builds Docker images for evaluation
- Supports `--push` flag to push images to registry
- Handles parallel builds with `--max-workers`

### README.md

- Brief description of the benchmark
- Setup instructions
- Usage examples with command-line invocation

## CLI Entrypoints

Register entrypoints in `pyproject.toml` under `[project.scripts]`:

```toml
[project.scripts]
<benchmark>-infer = "benchmarks.<benchmark>.run_infer:main"
<benchmark>-eval = "benchmarks.<benchmark>.eval_infer:main"
```

Use kebab-case for CLI names (e.g., `swebench-infer`).

## Naming Conventions

- **Benchmark names**: lowercase, hyphenated (e.g., `swebench`, `swtbench`)
- **Python modules**: lowercase with underscores (e.g., `benchmarks.swebench`)
- **Classes**: PascalCase (e.g., `SWEbenchEvaluation`)
- **Functions/methods**: snake_case (e.g., `prepare_instances`)
- **CLI arguments**: kebab-case (e.g., `--n-limit`)
- **Environment variables**: UPPER_SNAKE_CASE

## Error Handling

- **Fail fast on unrecoverable errors**: Raise exceptions rather than logging warnings when the error prevents evaluation.
- **Be lenient with recoverable errors**: A recoverable error (e.g., a single instance failing) should be logged but not crash the entire evaluation run.
- **Example**: Missing an optional field in one instance → log warning and skip. Missing a required field → raise an error.

## Testing

When adding a new benchmark, add tests to `tests/` following the pattern `test_<benchmark>_<feature>.py`:

```bash
tests/
├── test_<benchmark>_run_infer.py   # Tests for run_infer logic
├── test_<benchmark>_eval_infer.py  # Tests for eval_infer logic
└── test_<benchmark>_build_images.py # Tests for image building (if applicable)
```

## Pull Request Guidelines

- **Minimal changes**: A benchmark PR should not modify `utils/` unless absolutely necessary.
- **Describe all changes**: List every file changed and why.
- **Test locally**: Run `uv run pytest` before submitting.
- **Update documentation**: Update the benchmark's README.md if adding new features.

## Code Style

- Run `uv run pre-commit run --files <changed_files>` before committing
- Follow existing patterns in the codebase
- Use type hints for function parameters and return values
