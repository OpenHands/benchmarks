#!/usr/bin/env python3
"""
Utility script to configure LLM settings and save them to .config/ directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from openhands.sdk import LLM


CONFIG_DIR = Path.cwd() / ".config"
DEFAULT_CONFIG_NAME = "default"

# Common LLM providers and their typical models
PROVIDER_TEMPLATES = {
    "openai": {
        "description": "OpenAI (GPT-4, GPT-3.5, etc.)",
        "default_model": "gpt-4-turbo-preview",
        "models": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ],
        "base_url": None,
    },
    "anthropic": {
        "description": "Anthropic (Claude)",
        "default_model": "claude-sonnet-4-20250514",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "base_url": None,
    },
    "azure": {
        "description": "Azure OpenAI",
        "default_model": "gpt-4",
        "models": ["gpt-4", "gpt-35-turbo"],
        "base_url": None,
        "requires_api_version": True,
    },
    "ollama": {
        "description": "Ollama (Local LLMs)",
        "default_model": "llama2",
        "models": ["llama2", "mistral", "codellama", "phi"],
        "base_url": "http://localhost:11434",
    },
    "openrouter": {
        "description": "OpenRouter",
        "default_model": "openai/gpt-4-turbo-preview",
        "models": [
            "openai/gpt-4-turbo-preview",
            "anthropic/claude-3-opus",
            "google/gemini-pro",
        ],
        "base_url": "https://openrouter.ai/api/v1",
    },
    "custom": {
        "description": "Custom provider",
        "default_model": "",
        "models": [],
        "base_url": None,
    },
}


def ensure_config_dir() -> Path:
    """Ensure the .config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def get_config_path(name: str) -> Path:
    """Get the path to a configuration file."""
    return CONFIG_DIR / f"{name}.json"


def list_configs() -> None:
    """List all available LLM configurations."""
    if not CONFIG_DIR.exists():
        print("No configurations found. Create one with: configure-llm create")
        return

    configs = list(CONFIG_DIR.glob("*.json"))
    if not configs:
        print("No configurations found. Create one with: configure-llm create")
        return

    print("\nAvailable LLM configurations:")
    print("-" * 60)
    for config_file in sorted(configs):
        name = config_file.stem
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
            model = config_data.get("model", "Unknown")
            base_url = config_data.get("base_url", "Default")
            if base_url is None:
                base_url = "Default"
            print(f"  • {name:<20} Model: {model:<30} Base URL: {base_url}")
        except Exception as e:
            print(f"  • {name:<20} [Error reading config: {e}]")
    print("-" * 60)
    print(f"\nTo use a configuration:")
    print(f"  uv run swebench-infer --llm-config-path .config/<name>.json ...")


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]"
    prompt = f"{prompt}: "

    value = input(prompt).strip()
    if not value and default:
        return default
    return value


def get_bool_input(prompt: str, default: bool = False) -> bool:
    """Get boolean input from user."""
    default_str = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{default_str}]: ").strip().lower()

    if not value:
        return default
    return value in ["y", "yes", "true", "1"]


def interactive_create() -> Dict:
    """Interactively create an LLM configuration."""
    print("\n" + "=" * 60)
    print("LLM Configuration Wizard")
    print("=" * 60)

    # Select provider
    print("\nSelect LLM provider:")
    providers = list(PROVIDER_TEMPLATES.keys())
    for i, provider in enumerate(providers, 1):
        desc = PROVIDER_TEMPLATES[provider]["description"]
        print(f"  {i}. {provider:<15} - {desc}")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(providers)}) [1]: ").strip()
            if not choice:
                choice = "1"
            idx = int(choice) - 1
            if 0 <= idx < len(providers):
                provider = providers[idx]
                break
            print(f"Please enter a number between 1 and {len(providers)}")
        except ValueError:
            print("Please enter a valid number")

    template = PROVIDER_TEMPLATES[provider]

    # Model selection
    print(f"\nSelected provider: {template['description']}")
    if template["models"]:
        print("\nCommon models for this provider:")
        for model in template["models"]:
            print(f"  • {model}")

    model = get_user_input("\nEnter model name", template["default_model"])

    # API key
    print("\n⚠️  Note: The API key will NOT be stored in the config file.")
    print("   Set it as an environment variable: export LLM_API_KEY=your_key")
    need_api_key = get_bool_input("\nDoes this model require an API key?", default=True)

    # Base URL
    base_url = None
    if template.get("base_url") is not None:
        base_url = get_user_input("\nEnter base URL", template.get("base_url", ""))
        if base_url == "":
            base_url = None

    # API version (for Azure)
    api_version = None
    if template.get("requires_api_version"):
        api_version = get_user_input(
            "\nEnter API version (for Azure)", "2024-02-15-preview"
        )

    # Advanced settings
    print("\n" + "-" * 60)
    print("Advanced Settings (press Enter to use defaults)")
    print("-" * 60)

    temperature = get_user_input("\nTemperature (0.0-2.0)", "0.0")
    try:
        temperature = float(temperature)
    except ValueError:
        temperature = 0.0

    max_output_tokens = get_user_input("\nMax output tokens (or leave empty)", "")
    if max_output_tokens:
        try:
            max_output_tokens = int(max_output_tokens)
        except ValueError:
            max_output_tokens = None
    else:
        max_output_tokens = None

    timeout = get_user_input("\nHTTP timeout in seconds (or leave empty)", "")
    if timeout:
        try:
            timeout = int(timeout)
        except ValueError:
            timeout = None
    else:
        timeout = None

    # Build configuration
    config = {
        "model": model,
        "temperature": temperature,
    }

    if base_url:
        config["base_url"] = base_url

    if api_version:
        config["api_version"] = api_version

    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens

    if timeout:
        config["timeout"] = timeout

    # Additional provider-specific settings
    if provider == "ollama":
        config["custom_llm_provider"] = "ollama"
        if base_url:
            config["ollama_base_url"] = base_url

    print("\n" + "=" * 60)
    print("Configuration Summary:")
    print("=" * 60)
    print(json.dumps(config, indent=2))
    print("=" * 60)

    if need_api_key:
        print("\n⚠️  Remember to set your API key:")
        print("   export LLM_API_KEY=your_api_key_here")

    return config


def create_config(name: Optional[str] = None, interactive: bool = True) -> None:
    """Create a new LLM configuration."""
    ensure_config_dir()

    if interactive:
        config = interactive_create()

        if name is None:
            name = get_user_input(
                f"\nEnter configuration name", DEFAULT_CONFIG_NAME
            ).strip()
    else:
        print("Non-interactive mode not yet implemented. Use interactive mode.")
        sys.exit(1)

    if not name:
        name = DEFAULT_CONFIG_NAME

    # Validate the configuration using pydantic
    try:
        llm = LLM(**config)
        # Convert back to dict for saving (without api_key)
        config_to_save = llm.model_dump(
            exclude={"api_key", "aws_access_key_id", "aws_secret_access_key"},
            exclude_none=True,
        )
    except Exception as e:
        print(f"\n❌ Invalid configuration: {e}")
        sys.exit(1)

    config_path = get_config_path(name)

    if config_path.exists():
        overwrite = get_bool_input(
            f"\nConfiguration '{name}' already exists. Overwrite?", default=False
        )
        if not overwrite:
            print("Configuration not saved.")
            return

    # Save configuration
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"\n✅ Configuration saved to: {config_path}")
    print(f"\nTo use this configuration:")
    print(f"  uv run swebench-infer --llm-config-path {config_path} ...")


def delete_config(name: str) -> None:
    """Delete an LLM configuration."""
    config_path = get_config_path(name)

    if not config_path.exists():
        print(f"Configuration '{name}' not found.")
        sys.exit(1)

    confirm = get_bool_input(
        f"Are you sure you want to delete '{name}'?", default=False
    )
    if confirm:
        config_path.unlink()
        print(f"✅ Configuration '{name}' deleted.")
    else:
        print("Deletion cancelled.")


def show_config(name: str) -> None:
    """Show the contents of an LLM configuration."""
    config_path = get_config_path(name)

    if not config_path.exists():
        print(f"Configuration '{name}' not found.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"\nConfiguration: {name}")
    print("=" * 60)
    print(json.dumps(config, indent=2))
    print("=" * 60)
    print(f"\nPath: {config_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Configure LLM settings for OpenHands benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new configuration interactively
  uv run configure-llm create

  # Create a named configuration
  uv run configure-llm create --name gpt4

  # List all configurations
  uv run configure-llm list

  # Show a specific configuration
  uv run configure-llm show default

  # Delete a configuration
  uv run configure-llm delete old-config
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration")
    create_parser.add_argument(
        "--name", type=str, help="Configuration name (default: 'default')"
    )

    # List command
    subparsers.add_parser("list", help="List all configurations")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show a configuration")
    show_parser.add_argument("name", type=str, help="Configuration name")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a configuration")
    delete_parser.add_argument("name", type=str, help="Configuration name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "create":
        create_config(name=args.name)
    elif args.command == "list":
        list_configs()
    elif args.command == "show":
        show_config(args.name)
    elif args.command == "delete":
        delete_config(args.name)


if __name__ == "__main__":
    main()
