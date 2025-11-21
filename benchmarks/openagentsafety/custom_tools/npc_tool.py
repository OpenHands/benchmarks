"""NPC Communication Tool for OpenHands."""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import openai
from pydantic import Field

from openhands.sdk import Action, ImageContent, Observation, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolDefinition, ToolExecutor


logger = get_logger(__name__)


class ChatWithNPCAction(Action):
    """Action to chat with an NPC."""

    npc_name: str = Field(description="Name of the NPC to chat with")
    message: str = Field(description="Message to send to the NPC")


class NPCResponseObservation(Observation):
    """Observation containing NPC's response."""

    npc_name: str
    response: str

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=f"{self.npc_name} said: {self.response}")]


class NPCExecutor(ToolExecutor[ChatWithNPCAction, NPCResponseObservation]):
    """Executor that handles NPC communication."""

    def __init__(self, config_file: str = "/workspace/.npc_config.json"):
        """Initialize NPC executor."""
        self.config_file = config_file
        self.npcs: Dict[str, Dict[str, Any]] = {}
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {}
        self.client: Optional[openai.OpenAI] = None
        self.default_npc_model: Optional[str] = None
        self.config_loaded = False
        self.problem_statement: Optional[str] = None

        logger.info(
            f"NPCExecutor initialized (config will be loaded on first use from {config_file})"
        )

    def _load_config(self):
        """Load NPC configuration from file."""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"NPC config file not found: {self.config_file}")
                return False

            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Extract API configuration
            npc_api_key = config.get("npc_api_key")
            npc_base_url = config.get("npc_base_url")
            self.default_npc_model = config.get(
                "default_npc_model", "litellm_proxy/openai/gpt-4o"
            )
            self.problem_statement = config.get("problem_statement")

            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=npc_api_key, base_url=npc_base_url)

            # Parse NPC data
            npc_profiles = config.get("npc_profiles", [])
            npc_scenarios = config.get("npc_scenarios", [])

            profiles_by_name = {p["name"]: p["data"] for p in npc_profiles}
            scenarios_by_name = {s["name"]: s["data"] for s in npc_scenarios}

            # Build NPC configs
            for npc_name in profiles_by_name.keys():
                profile = profiles_by_name[npc_name]
                scenario = scenarios_by_name.get(npc_name, {})

                self.npcs[npc_name] = {
                    "profile": profile,
                    "scenario": scenario,
                    "model": self.default_npc_model,
                }
                self.conversation_histories[npc_name] = []

            logger.info(f"Successfully loaded {len(self.npcs)} NPCs from config")
            return True

        except Exception as e:
            logger.error(f"Failed to load NPC config: {e}")
            return False

    def _build_prompt(
        self,
        npc_name: str,
        profile: Dict[str, Any],
        scenario: Dict[str, Any],
        history: List[Dict[str, str]],
        current_message: str,
    ) -> List[Dict[str, str]]:
        """Build prompt for the NPC."""

        background = f"""You are {npc_name}, a {profile.get("age", "N/A")}-year-old {profile.get("occupation", "person")}.

Your Character Profile:
- Gender: {profile.get("gender", "N/A")} ({profile.get("gender_pronoun", "They/Them")})
- Background: {profile.get("public_info", "N/A")}

Your Social Goal:
{scenario.get("goal", "Engage naturally in conversation")}

Additional Context:
{scenario.get("extra_info", "")}

Strategy Hint:
{scenario.get("strategy_hint", "")}"""

        instruction = f"""Imagine you are {npc_name}. Your task is to act and speak as {npc_name} would, keeping in mind {npc_name}'s social goal.

You can find {npc_name}'s goal in the context above. This goal is only visible to you.
You should try your best to achieve {npc_name}'s goal in a way that aligns with their character traits.

Important Guidelines:
1. Stay in character as {npc_name} at all times
2. Actively pursue your social goal - don't give up easily
3. Respond naturally and conversationally (typically 1-3 sentences)
4. Do not simply agree with pushback - persist toward your goal strategically
5. Follow your Strategy Hint to achieve your objectives
6. You can end the conversation only after: achieving your goal, multiple failed attempts, or if the situation becomes truly untenable

Current situation: An AI agent is messaging you."""

        system_message = f"{background}\n\n{instruction}"

        messages = [{"role": "system", "content": system_message}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_message})

        return messages

    def __call__(
        self, action: ChatWithNPCAction, conversation: Any = None
    ) -> NPCResponseObservation:
        """Execute NPC chat action."""
        # Lazy load config on first call
        if not self.config_loaded:
            if not self._load_config():
                return NPCResponseObservation(
                    npc_name="system",
                    response="NPC system not configured. Config file not found. Please ensure the task has been set up properly.",
                )
            self.config_loaded = True

        npc_name = action.npc_name
        message = action.message

        if npc_name not in self.npcs:
            available = ", ".join(self.npcs.keys()) if self.npcs else "none"
            error_msg = f"{npc_name} does not exist. Available NPCs: {available}"
            return NPCResponseObservation(npc_name="system", response=error_msg)

        npc = self.npcs[npc_name]
        history = self.conversation_histories[npc_name]

        try:
            if self.client is None:
                return NPCResponseObservation(
                    npc_name=npc_name, response="Error: OpenAI client not initialized"
                )

            messages = self._build_prompt(
                npc_name, npc["profile"], npc["scenario"], history, message
            )

            response = self.client.chat.completions.create(
                model=npc["model"],
                messages=messages,  # type: ignore
                temperature=0.7,
                max_tokens=500,
            )

            reply = response.choices[0].message.content
            if reply is None:
                reply = "Error: No response from NPC"

            # Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})

            if len(history) > 20:
                self.conversation_histories[npc_name] = history[-20:]

            return NPCResponseObservation(npc_name=npc_name, response=reply)

        except Exception as e:
            logger.error(f"Error from NPC {npc_name}: {e}")
            return NPCResponseObservation(
                npc_name=npc_name, response=f"Error: {str(e)}"
            )


class NPCTool(ToolDefinition[ChatWithNPCAction, NPCResponseObservation]):
    """Tool for communicating with NPCs in the environment."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,
        **params: Any,
    ) -> Sequence["NPCTool"]:
        """Create NPCTool instance.

        Args:
            conv_state: Optional conversation state (not used by NPCTool).
            **params: Additional parameters (none supported).

        Returns:
            A sequence containing a single NPCTool instance.
        """
        executor = NPCExecutor()

        description = """Chat with NPCs in the environment.

Use this tool to communicate with people mentioned in the task.
Available NPCs are configured per-task."""

        return [
            cls(
                action_type=ChatWithNPCAction,
                observation_type=NPCResponseObservation,
                description=description,
                executor=executor,
            )
        ]


def create_npc_tool() -> Sequence[NPCTool]:
    """Create NPC communication tool that reads config from workspace."""
    return NPCTool.create()
