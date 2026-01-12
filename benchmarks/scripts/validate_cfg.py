import argparse

from benchmarks.utils.llm_config import load_llm_config


def main():
    parser = argparse.ArgumentParser(description="Validate LLM configuration")
    parser.add_argument("config_path", type=str, help="Path to JSON LLM configuration")
    args = parser.parse_args()

    llm = load_llm_config(args.config_path)

    print("LLM configuration is valid:")
    print(llm.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
