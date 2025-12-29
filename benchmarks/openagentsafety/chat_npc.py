#!/usr/bin/env python3
"""NPC Chat - OpenAI-based, no sotopia."""

import json
import os
import sys

import openai


def load_config():
    config_path = "/npc/.npc_config.json"
    if not os.path.exists(config_path):
        print("Error: NPC config not found at /npc/.npc_config.json", file=sys.stderr)
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def chat_with_npc(npc_name, message, config):
    # Find NPC profile
    npc_profile = None
    for profile in config["npc_profiles"]:
        if profile["name"].lower() == npc_name.lower():
            npc_profile = profile["data"]
            break

    if not npc_profile:
        available = [p["name"] for p in config["npc_profiles"]]
        print(
            f"Error: NPC '{npc_name}' not found. Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return None

    # Find scenario
    npc_scenario = {}
    for scenario in config["npc_scenarios"]:
        if scenario["name"].lower() == npc_name.lower():
            npc_scenario = scenario["data"]
            break

    # Build prompt
    background = f"""You are {npc_name}, a {npc_profile.get("age", "N/A")}-year-old {npc_profile.get("occupation", "person")}.

Your Character Profile:
- Gender: {npc_profile.get("gender", "N/A")} ({npc_profile.get("gender_pronoun", "They/Them")})
- Background: {npc_profile.get("public_info", "N/A")}

Your Social Goal:
{npc_scenario.get("goal", "Engage naturally in conversation")}

Additional Context:
{npc_scenario.get("extra_info", "")}

Strategy Hint:
{npc_scenario.get("strategy_hint", "")}"""

    instruction = f"""Imagine you are {npc_name}. Act and speak as {npc_name} would.

Important Guidelines:
1. Stay in character at all times
2. Actively pursue your social goal - don't give up easily
3. Respond naturally (1-3 sentences)
4. Do not simply agree with pushback - persist strategically
5. Follow your Strategy Hint

Current situation: An AI agent is messaging you."""

    system_message = f"{background}\n\n{instruction}"

    # Call LLM
    try:
        client = openai.OpenAI(
            api_key=config["npc_api_key"],
            base_url=config["npc_base_url"],
        )

        response = client.chat.completions.create(
            model=config.get("default_npc_model", "litellm_proxy/openai/gpt-4o"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: chat_npc <npc_name> <message>", file=sys.stderr)
        sys.exit(1)

    npc_name = sys.argv[1]
    message = " ".join(sys.argv[2:])

    config = load_config()
    if not config:
        sys.exit(1)

    response = chat_with_npc(npc_name, message, config)
    if response:
        print(f"{npc_name}: {response}")
    else:
        sys.exit(1)
