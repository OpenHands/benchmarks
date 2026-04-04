from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PARALLELISM_LEVELS = [1, 2, 4, 8, 16, 32]
DEFAULT_CONTEXT_LENGTHS_K = [32, 128]
DEFAULT_MACHINE_SIZES = ["4xh100", "8xh100"]
DEFAULT_MAX_ITERATIONS = 80
DEFAULT_AGENT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_MAX_FAKE_RESPONSES = 8
DEFAULT_WORKSPACE_ROOT = "/tmp"
DEFAULT_OUTPUT_DIR = "./eval_outputs"
DEFAULT_COLLAPSE_FAILURE_RATE = 1.0
VLLM_PORT = 8000

DEFAULT_TASK_PROMPT = """\
In directory {workspace_dir}, create a simple and modular library for generating an HTML page from Python documentation, and apply the library to itself.

Requirements:
- Build the library under src/pydoc_html/.
- Create a small CLI or module entrypoint that can render a package's Python docstrings into a single HTML page.
- Apply the library to itself and write the generated page to site/index.html.
- Keep the code simple, modular, and easy to understand.
- When you are done, use the finish tool and summarize what you created.
"""


@dataclass(frozen=True)
class VLLMModelRecipe:
    alias: str
    model_name: str
    served_model_name: str
    tensor_parallel_4gpu: int
    tensor_parallel_8gpu: int
    enable_expert_parallel_4gpu: bool = False
    enable_expert_parallel_8gpu: bool = False
    trust_remote_code: bool = True
    kv_cache_dtype: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    compilation_config: str | None = None
    extra_env: tuple[tuple[str, str], ...] = ()
    model_revision: str | None = None


MODEL_RECIPES: dict[str, VLLMModelRecipe] = {
    "nemotron-3-super": VLLMModelRecipe(
        alias="nemotron-3-super",
        model_name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        served_model_name="nemotron-3-super",
        tensor_parallel_4gpu=4,
        tensor_parallel_8gpu=8,
        kv_cache_dtype="fp8",
        tool_call_parser="qwen3_coder",
        reasoning_parser="nemotron_v3",
        extra_env=(("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),),
    ),
    "minimax-m2.5": VLLMModelRecipe(
        alias="minimax-m2.5",
        model_name="MiniMaxAI/MiniMax-M2.5",
        served_model_name="minimax-m2.5",
        tensor_parallel_4gpu=4,
        tensor_parallel_8gpu=8,
        enable_expert_parallel_8gpu=True,
        tool_call_parser="minimax_m2",
        reasoning_parser="minimax_m2_append_think",
        trust_remote_code=True,
    ),
}

MACHINE_GPU_COUNTS = {"4xh100": 4, "8xh100": 8}


def get_model_recipe(alias: str) -> VLLMModelRecipe:
    try:
        return MODEL_RECIPES[alias]
    except KeyError as exc:
        options = ", ".join(sorted(MODEL_RECIPES))
        raise ValueError(
            f"Unknown model alias {alias!r}. Expected one of: {options}"
        ) from exc


def get_gpu_count(machine_size: str) -> int:
    try:
        return MACHINE_GPU_COUNTS[machine_size]
    except KeyError as exc:
        options = ", ".join(sorted(MACHINE_GPU_COUNTS))
        raise ValueError(
            f"Unknown machine size {machine_size!r}. Expected one of: {options}"
        ) from exc


def context_length_tokens(context_length_k: int) -> int:
    return context_length_k * 1024
