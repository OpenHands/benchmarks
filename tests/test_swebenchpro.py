import json

import pandas as pd

from benchmarks.swebenchpro.build_images import (
    collect_unique_base_images,
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.swebenchpro.constants import SOURCE_REPO_PATH
from benchmarks.swebenchpro.eval_infer import (
    convert_to_swebenchpro_format,
    write_report,
)


def test_get_official_docker_image_uses_dataset_dockerhub_tag():
    image = get_official_docker_image(
        {
            "instance_id": "instance_demo",
            "dockerhub_tag": "nodebb.nodebb-instance_demo",
        }
    )

    assert image == "docker.io/jefzda/sweap-images:nodebb.nodebb-instance_demo"
    assert extract_custom_tag(image) == "nodebb.nodebb-instance_demo"


def test_collect_unique_base_images_deduplicates(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "instance_id": "instance_a",
                "dockerhub_tag": "repo-image-a",
            },
            {
                "instance_id": "instance_b",
                "dockerhub_tag": "repo-image-a",
            },
            {
                "instance_id": "instance_c",
                "dockerhub_tag": "repo-image-c",
            },
        ]
    )
    monkeypatch.setattr(
        "benchmarks.swebenchpro.build_images.get_dataset",
        lambda dataset_name, split, eval_limit, selected_instances_file: df,
    )

    images = collect_unique_base_images("ScaleAI/SWE-bench_Pro", "test", 0)

    assert images == [
        "docker.io/jefzda/sweap-images:repo-image-a",
        "docker.io/jefzda/sweap-images:repo-image-c",
    ]


def test_extract_custom_tag_shortens_long_tags():
    image = (
        "docker.io/jefzda/sweap-images:"
        "qutebrowser.qutebrowser-qutebrowser__qutebrowser-"
        "5fdc83e5da6222fe61163395baaad7ae57fa2cb4-v363c8a7e5ccdf6968fc7ab84a2053ac780366"
    )

    custom_tag = extract_custom_tag(image)

    assert len(custom_tag) <= 100
    assert custom_tag.startswith("qutebrowser.qutebrowser-qutebrowser__qutebrowser-")
    assert custom_tag != image.rsplit(":", 1)[1]


def test_convert_to_swebenchpro_format_writes_patch_array(tmp_path):
    input_path = tmp_path / "output.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "instance_a",
                        "test_result": {"git_patch": "diff --git a/a b/a"},
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "instance_b",
                        "test_result": {},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output.swebenchpro.json"

    convert_to_swebenchpro_format(
        str(input_path), str(output_path), prefix="demo-model"
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == [
        {
            "instance_id": "instance_a",
            "patch": "diff --git a/a b/a",
            "prefix": "demo-model",
        },
        {
            "instance_id": "instance_b",
            "patch": "",
            "prefix": "demo-model",
        },
    ]


def test_write_report_records_resolved_ids(tmp_path):
    eval_results_path = tmp_path / "eval_results.json"
    eval_results_path.write_text(
        json.dumps(
            {
                "instance_a": True,
                "instance_b": False,
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "output.report.json"

    report = write_report(eval_results_path, report_path)

    assert report["resolved_instances"] == 1
    assert report["unresolved_instances"] == 1
    assert report["resolved_ids"] == ["instance_a"]
    assert report["unresolved_ids"] == ["instance_b"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_source_repo_path_constant_matches_swebench_pro_layout():
    assert SOURCE_REPO_PATH == "/app"
