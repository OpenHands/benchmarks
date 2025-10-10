import argparse

from openhands.sdk import LLM


def main():
    parser = argparse.ArgumentParser(description="Validate LLM configuration")
    parser.add_argument("config_path", type=str, help="Path to JSON LLM configuration")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)

    print("LLM configuration is valid:")
    print(llm.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
