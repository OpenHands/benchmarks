from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

import httpx
import modal


try:
    from benchmarks.agentserving.config import (
        VLLM_PORT,
        context_length_tokens as _context_length_tokens,
        get_gpu_count as _get_gpu_count,
        get_model_recipe as _get_model_recipe,
    )

    def get_model_recipe(alias: str) -> Any:
        return _get_model_recipe(alias)

    def get_gpu_count(machine_size: str) -> int:
        return _get_gpu_count(machine_size)

    def context_length_tokens(context_length_k: int) -> int:
        return _context_length_tokens(context_length_k)

except ModuleNotFoundError:
    VLLM_PORT = 8000

    @dataclass(frozen=True)
    class _VLLMModelRecipe:
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

    _MODEL_RECIPES = {
        "nemotron-3-super": _VLLMModelRecipe(
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
        "minimax-m2.5": _VLLMModelRecipe(
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
    _GPU_COUNTS = {"4xh100": 4, "8xh100": 8}

    def get_model_recipe(alias: str) -> Any:
        return _MODEL_RECIPES[alias]

    def get_gpu_count(machine_size: str) -> int:
        return _GPU_COUNTS[machine_size]

    def context_length_tokens(context_length_k: int) -> int:
        return context_length_k * 1024


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


MODEL_ALIAS = os.getenv("AGENTSERVING_MODEL_ALIAS", "nemotron-3-super")
MACHINE_SIZE = os.getenv("AGENTSERVING_MACHINE_SIZE", "4xh100")
CONTEXT_LENGTH_K = int(os.getenv("AGENTSERVING_CONTEXT_LENGTH_K", "32"))
FAST_BOOT = _env_flag("AGENTSERVING_FAST_BOOT", default=False)
PIECEWISE_CUDAGRAPH = _env_flag("AGENTSERVING_PIECEWISE_CUDAGRAPH", default=False)
recipe = get_model_recipe(MODEL_ALIAS)
GPU_COUNT = get_gpu_count(MACHINE_SIZE)
APP_NAME = os.getenv(
    "AGENTSERVING_APP_NAME",
    f"agentserving-{MODEL_ALIAS.replace('.', '-')}-{MACHINE_SIZE}-{CONTEXT_LENGTH_K}k",
)
VLLM_VERSION = os.getenv("AGENTSERVING_VLLM_VERSION", "0.19.0")


app = modal.App(APP_NAME)

hf_cache_volume = modal.Volume.from_name(
    "agentserving-hf-cache", create_if_missing=True
)
vllm_cache_volume = modal.Volume.from_name(
    "agentserving-vllm-cache", create_if_missing=True
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        "transformers>=4.57.1",
        "huggingface_hub[hf_transfer]>=0.34.0",
        "httpx>=0.27.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "AGENTSERVING_MODEL_ALIAS": MODEL_ALIAS,
            "AGENTSERVING_MACHINE_SIZE": MACHINE_SIZE,
            "AGENTSERVING_CONTEXT_LENGTH_K": str(CONTEXT_LENGTH_K),
            "AGENTSERVING_FAST_BOOT": "1" if FAST_BOOT else "0",
            "AGENTSERVING_PIECEWISE_CUDAGRAPH": "1" if PIECEWISE_CUDAGRAPH else "0",
        }
    )
)


def _build_secret() -> modal.Secret | None:
    values: dict[str, str | None] = {
        key: value
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        if (value := os.getenv(key))
    }
    if not values:
        return None
    return modal.Secret.from_dict(values)


secrets = [secret] if (secret := _build_secret()) is not None else []


def build_vllm_command() -> list[str]:
    context_tokens = context_length_tokens(CONTEXT_LENGTH_K)
    tensor_parallel = (
        recipe.tensor_parallel_4gpu if GPU_COUNT == 4 else recipe.tensor_parallel_8gpu
    )
    enable_expert_parallel = (
        recipe.enable_expert_parallel_4gpu
        if GPU_COUNT == 4
        else recipe.enable_expert_parallel_8gpu
    )

    command = [
        "vllm",
        "serve",
        recipe.model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--served-model-name",
        recipe.served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--max-model-len",
        str(context_tokens),
        "--max-log-len",
        "0",
        "--uvicorn-log-level",
        "info",
        "--enable-auto-tool-choice",
    ]
    if recipe.model_revision:
        command.extend(["--revision", recipe.model_revision])
    if recipe.trust_remote_code:
        command.append("--trust-remote-code")
    if recipe.kv_cache_dtype:
        command.extend(["--kv-cache-dtype", recipe.kv_cache_dtype])
    if recipe.tool_call_parser:
        command.extend(["--tool-call-parser", recipe.tool_call_parser])
    if recipe.reasoning_parser:
        command.extend(["--reasoning-parser", recipe.reasoning_parser])
    if enable_expert_parallel:
        command.append("--enable-expert-parallel")
    if FAST_BOOT:
        command.append("--enforce-eager")
    else:
        command.append("--no-enforce-eager")
    if PIECEWISE_CUDAGRAPH or recipe.compilation_config:
        compilation_config = recipe.compilation_config or json.dumps(
            {"cudagraph_mode": "PIECEWISE"}
        )
        command.extend(["--compilation-config", compilation_config])
    return command


@app.function(
    image=image,
    gpu=f"H100:{GPU_COUNT}",
    scaledown_window=15 * 60,
    timeout=2 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=secrets,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=VLLM_PORT, startup_timeout=2 * 60 * 60)
def serve() -> None:
    env = os.environ.copy()
    for key, value in recipe.extra_env:
        env[key] = value
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    command = build_vllm_command()
    print("Launching vLLM:")
    print(shlex.join(command))
    subprocess.Popen(command, env=env)


@app.local_entrypoint()
def show_config() -> None:
    print(
        json.dumps(
            {
                "app_name": APP_NAME,
                "model_alias": MODEL_ALIAS,
                "machine_size": MACHINE_SIZE,
                "context_length_k": CONTEXT_LENGTH_K,
                "gpu_count": GPU_COUNT,
                "vllm_version": VLLM_VERSION,
                "command": build_vllm_command(),
            },
            indent=2,
        )
    )
    if _env_flag("AGENTSERVING_PROBE_HEALTH", default=False):
        url = serve.get_web_url()
        response = httpx.get(f"{url}/health", timeout=30.0)
        response.raise_for_status()
        print(f"Server is healthy at {url}")
