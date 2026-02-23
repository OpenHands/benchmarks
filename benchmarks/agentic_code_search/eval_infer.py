import json
from argparse import ArgumentParser


filtered_instances = [
    "pytest-dev__pytest-5227",
    "sympy__sympy-15345",
    "sympy__sympy-21614",
    "scikit-learn__scikit-learn-13439",
    "sympy__sympy-11400",
    "sympy__sympy-19487",
    "sympy__sympy-15308",
    "django__django-12915",
    "sympy__sympy-20590",
    "sympy__sympy-17022",
    "django__django-11099",
    "django__django-13220",
    "django__django-11964",
    "matplotlib__matplotlib-25332",
    "django__django-10914",
    "django__django-14915",
    "django__django-11049",
    "django__django-11564",
    "sympy__sympy-17655",
    "sympy__sympy-16106",
    "sympy__sympy-12171",
    "django__django-15400",
    "django__django-14411",
    "sympy__sympy-21055",
    "django__django-15213",
    "django__django-15902",
]


def main(args):
    results_file = args.results_file
    f1_file = 0
    f1_function = 0
    f1_module = 0
    num_steps = 0
    num_tool_calls = 0
    total_time = 0
    cnt = 0
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            if (
                result["instance_id"] in filtered_instances
                and "SWE-bench_Lite" in results_file
            ):
                continue
            test_result = result["test_result"]
            if "num_steps" in test_result:
                num_steps += test_result["num_steps"]
            if "num_tool_calls" in test_result:
                num_tool_calls += test_result["num_tool_calls"]
            if "wall_time_seconds" in test_result:
                total_time += test_result["wall_time_seconds"]

            reward_dict = result["test_result"]["reward"]
            cnt += 1
            if reward_dict is not None:
                f1_file += reward_dict.get("file_reward", 0)
                f1_module += reward_dict.get("module_reward", 0)
                f1_function += reward_dict.get("entity_reward", 0)
    # cnt = 266
    print(f"Average File F1 score: {f1_file / cnt:.4f} over {cnt} samples")
    print(f"Average Module F1 score: {f1_module / cnt:.4f} over {cnt} samples")
    print(f"Average Function F1 score: {f1_function / cnt:.4f} over {cnt} samples")
    print(f"Average # of steps: {num_steps / cnt:.4f} over {cnt} samples")
    print(f"Average # of tool calls: {num_tool_calls / cnt:.4f} over {cnt} samples")
    print(f"Average wall time (s): {total_time / cnt:.4f} over {cnt} samples")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
