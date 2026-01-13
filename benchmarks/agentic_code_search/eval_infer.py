import json
from argparse import ArgumentParser


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
