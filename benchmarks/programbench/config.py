"""ProgramBench configuration defaults.

ProgramBench (https://programbench.com / https://github.com/facebookresearch/ProgramBench)
ships its task metadata inside the upstream ``programbench`` PyPI package and the
per-task Docker images under the ``programbench`` Docker Hub org with the
``task_cleanroom`` tag (e.g. ``programbench/abishekvashok_1776_cmatrix.5c082c6:task_cleanroom``).
"""

from typing import TypedDict


class _InferDefaults(TypedDict):
    dataset: str
    split: str
    output_dir: str
    num_workers: int
    workspace_dir: str
    task_image_tag: str
    build_target: str
    max_iterations: int


class _EvalDefaults(TypedDict):
    image_tag: str
    workers: int
    branch_workers: int
    docker_cpus: int


# Default inference settings (only include values actually used by argparse).
INFER_DEFAULTS: _InferDefaults = {
    # ProgramBench has a single canonical 200-task split shipped with the
    # ``programbench`` package. We expose this purely as a label that ends up
    # in the structured output dir name.
    "dataset": "programbench/ProgramBench",
    "split": "test",
    "output_dir": "./eval_outputs",
    "num_workers": 1,
    # Submission tarballs default to using the agent's /workspace contents.
    "workspace_dir": "/workspace",
    # The cleanroom task image tag used for inference. ProgramBench tags
    # cleanroom variants with ``task_cleanroom`` (binary + docs only, no
    # internet). Other tags exist on Docker Hub but should not be used for
    # inference.
    "task_image_tag": "task_cleanroom",
    # Build target for layering openhands-agent-server on top of the
    # cleanroom image. ``source-minimal`` keeps the image small.
    "build_target": "source-minimal",
    # Conversation iteration budget. ProgramBench tasks tend to be larger
    # than typical SWE-Bench instances since the agent rebuilds an entire
    # codebase from scratch. We allow up to 1000 because:
    #   1. The full rebuild loop (read docs → infer interface → implement
    #      → run probes against the reference binary → diagnose failures
    #      → patch) can chain many bash + edit + test iterations on
    #      non-trivial CLIs.
    #   2. Stop hooks (compile-contract + reference-diffs) may reject
    #      the agent's first attempt to finish and demand more work; we
    #      want a generous budget for those retries.
    # Lowering this for cost only makes sense for quick smoke runs (pass
    # ``--max-iterations`` explicitly).
    "max_iterations": 1000,
}

# Default evaluation settings.
EVAL_DEFAULTS: _EvalDefaults = {
    # Image tag used by ``programbench eval`` to spin up evaluation
    # containers. ``task`` is the upstream default and corresponds to the
    # full task image (binary + tests scaffolding).
    "image_tag": "task",
    # Parallelism for ``programbench eval``.
    "workers": 1,
    "branch_workers": 1,
    # CPU cores allotted per docker container during evaluation. Mirrors
    # programbench.constants.DOCKER_CPUS default (10).
    "docker_cpus": 10,
}
