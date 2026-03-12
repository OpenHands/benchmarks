from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from benchmarks.utils.build_utils import BuildOutput, build_all_images
from benchmarks.utils.buildx_utils import (
    BackgroundBuildKitPruner,
    buildkit_prune_filters_for_completed_images,
)


class FakeFuture:
    def __init__(self, result: BuildOutput):
        self._result = result

    def result(self) -> BuildOutput:
        return self._result

    def __hash__(self) -> int:
        return id(self)


class FakeExecutor:
    def __init__(self, results: dict[str, BuildOutput], *args, **kwargs):
        self._results = results

    def __enter__(self) -> FakeExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> FakeFuture:
        return FakeFuture(self._results[kwargs["base_image"]])


class FakeTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self) -> FakeTqdm:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, n: int = 1) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass

    def set_postfix_str(self, postfix: str, refresh: bool = True) -> None:
        pass


class FakePruner:
    instances: list[FakePruner] = []

    def __init__(self, *, enabled: bool, timeout_sec: int):
        self.enabled = enabled
        self.timeout_sec = timeout_sec
        self.enqueued: list[list[str]] = []
        self.wait_called = False
        FakePruner.instances.append(self)

    @property
    def is_busy(self) -> bool:
        return False

    def enqueue_completed_batch(self, base_images: list[str]) -> None:
        self.enqueued.append(list(base_images))

    def poll(self) -> None:
        pass

    def wait(self) -> None:
        self.wait_called = True


def test_buildkit_prune_filters_for_completed_images_escapes_values() -> None:
    filters = buildkit_prune_filters_for_completed_images(
        [
            "docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-18199:latest",
            "docker.io/acme/repo.with.dots:latest",
        ]
    )

    assert filters == [
        "description~=docker\\.io/swebench/sweb\\.eval\\.x86_64\\.sympy_1776_sympy\\-18199:latest|docker\\.io/acme/repo\\.with\\.dots:latest"
    ]


def test_background_buildkit_pruner_processes_batches_in_order() -> None:
    calls: list[tuple[int | None, list[str] | None, int | None]] = []

    def fake_prune_buildkit_cache(
        *,
        keep_storage_gb: int | None = None,
        filters: list[str] | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        calls.append((keep_storage_gb, filters, timeout_sec))

    with patch(
        "benchmarks.utils.buildx_utils.prune_buildkit_cache",
        side_effect=fake_prune_buildkit_cache,
    ):
        pruner = BackgroundBuildKitPruner(enabled=True, timeout_sec=17)
        pruner.enqueue_completed_batch(["base-1", "base-2"])
        pruner.enqueue_completed_batch(["base-3"])
        pruner.wait()

    assert calls == [
        (None, buildkit_prune_filters_for_completed_images(["base-1", "base-2"]), 17),
        (None, buildkit_prune_filters_for_completed_images(["base-3"]), 17),
    ]


@patch("benchmarks.utils.build_utils.tqdm", FakeTqdm)
def test_build_all_images_enqueues_previous_successful_batch_for_pruning(
    tmp_path: Path,
) -> None:
    FakePruner.instances.clear()
    results = {
        "base-1": BuildOutput(base_image="base-1", tags=["tag-1"], error=None),
        "base-2": BuildOutput(base_image="base-2", tags=[], error="boom"),
        "base-3": BuildOutput(base_image="base-3", tags=["tag-3"], error=None),
        "base-4": BuildOutput(base_image="base-4", tags=["tag-4"], error=None),
    }

    with (
        patch(
            "benchmarks.utils.build_utils.ProcessPoolExecutor",
            side_effect=lambda *args, **kwargs: FakeExecutor(results, *args, **kwargs),
        ),
        patch(
            "benchmarks.utils.build_utils.as_completed",
            side_effect=lambda futures: list(futures),
        ),
        patch("benchmarks.utils.build_utils.BackgroundBuildKitPruner", FakePruner),
        patch(
            "benchmarks.utils.build_utils.maybe_prune_buildkit_cache",
            return_value=False,
        ) as mock_prune,
        patch("benchmarks.utils.build_utils.buildkit_disk_usage", return_value=(0, 0)),
        patch.dict(
            os.environ,
            {
                "BUILD_BATCH_SIZE": "2",
                "BUILDKIT_COMPLETED_BATCH_PRUNE_ENABLED": "1",
            },
            clear=False,
        ),
    ):
        exit_code = build_all_images(
            base_images=["base-1", "base-2", "base-3", "base-4"],
            target="source-minimal",
            build_dir=tmp_path,
            max_workers=2,
            max_retries=1,
        )

    assert exit_code == 1
    assert mock_prune.call_count == 2

    pruner = FakePruner.instances[0]
    assert pruner.wait_called is True
    assert pruner.enqueued == [["base-1"], ["base-3", "base-4"]]
