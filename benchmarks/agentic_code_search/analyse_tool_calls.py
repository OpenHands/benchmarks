import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt


def main(args):
    results_file = args.results_file
    tool_call_counts = {}
    tot = 0
    with open(results_file, "r") as f:
        for line in f:
            trajectory = json.loads(line)
            cnt = 0
            for event in trajectory.get("history", []):
                if event["kind"] == "ActionEvent" and event["source"] == "agent":
                    cnt += 1
            if cnt == 0:
                continue
            tool_call_counts[cnt] = tool_call_counts.get(cnt, 0) + 1
            tot += 1
    print(tool_call_counts)
    for k, v in tool_call_counts.items():
        tool_call_counts[k] = (v / tot) * 100
    # plot histogram
    plt.bar(list(tool_call_counts.keys()), list(tool_call_counts.values()))
    plt.xlabel("Number of Tool Calls")
    plt.ylabel("Percentage of Instances (%)")
    plt.savefig(args.output_image)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the JSONL file containing eventstream data.",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="tool_call_histogram.png",
        help="Path to save the output histogram image.",
    )
    args = parser.parse_args()
    main(args)
