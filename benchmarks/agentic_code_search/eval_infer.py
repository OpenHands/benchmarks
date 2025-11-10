import json
from argparse import ArgumentParser


def main(args):
    results_file = args.results_file
    f1_total = 0
    cnt = 0
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            print(result["test_result"]["reward"])
            if result["test_result"]["reward"] is not None:
                f1_total += result["test_result"]["reward"]
            cnt += 1
    print(f"Average F1 score: {f1_total / cnt:.4f} over {cnt} samples")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
