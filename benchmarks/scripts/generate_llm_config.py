from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_config(
    model: str,
    output_path: str,
    api_base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> None:
    llm_config: dict[str, object] = {"model": model}

    if api_base_url:
        llm_config["base_url"] = api_base_url
    if api_key_env:
        llm_config["api_key_env"] = api_key_env
    if temperature is not None:
        llm_config["temperature"] = temperature
    if top_p is not None:
        llm_config["top_p"] = top_p
    if max_completion_tokens is not None:
        llm_config["max_output_tokens"] = max_completion_tokens
    if timeout is not None:
        llm_config["timeout"] = timeout
    if max_retries is not None:
        llm_config["num_retries"] = max_retries

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(llm_config, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote LLM config to {str(out_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM config from CLI args",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Model name/id")
    parser.add_argument("--api-base-url", type=str, help="API base URL")
    parser.add_argument(
        "--api-key-env",
        type=str,
        help="Environment variable name containing the API key",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling (top-p)")
    parser.add_argument("--max-completion-tokens", type=int, help="Max completion tokens")
    parser.add_argument("--timeout", type=int, help="API timeout in seconds")
    parser.add_argument("--max-retries", type=int, help="Max API call retries")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to write the generated JSON config",
    )

    args = parser.parse_args()

    generate_config(
        model=args.model,
        output_path=args.output_path,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
