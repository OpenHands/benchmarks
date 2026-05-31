"""Harbor dataset names for benchmarks with Harbor-backed wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


HARBOR_DATASET_BY_BENCHMARK: Mapping[str, str] = MappingProxyType(
    {
        "gaia": "gaia",
        "multiswebench": "multi-swe-bench",
        "skillsbench": "benchflow/skillsbench",
        "swebench": "swebench-verified",
        "swebenchmultilingual": "swebench_multilingual",
        "swebenchpro": "swebenchpro",
        "swesmith": "swesmith",
        "swtbench": "swtbench-verified",
        "swegym": "swegym",
        "terminalbench": "terminal-bench@2.0",
    }
)


def normalize_benchmark_name(benchmark_name: str) -> str:
    """Normalize benchmark names to the package/module key used in this repo."""
    return (
        benchmark_name.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
    )


def is_harbor_covered(benchmark_name: str) -> bool:
    """Return whether the benchmark has a Harbor dataset mapping."""
    return normalize_benchmark_name(benchmark_name) in HARBOR_DATASET_BY_BENCHMARK


def get_harbor_dataset(benchmark_name: str) -> str:
    """Return the Harbor dataset name for a covered benchmark."""
    key = normalize_benchmark_name(benchmark_name)
    try:
        return HARBOR_DATASET_BY_BENCHMARK[key]
    except KeyError as exc:
        supported = ", ".join(sorted(HARBOR_DATASET_BY_BENCHMARK))
        raise KeyError(
            f"No Harbor dataset mapping for {benchmark_name!r}. "
            f"Supported benchmarks: {supported}"
        ) from exc
