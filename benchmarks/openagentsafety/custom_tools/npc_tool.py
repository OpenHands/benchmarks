"""NPC Communication Tool for OpenHands."""

import openai
import json
import os
from typing import Dict, Any, List, Sequence
from pydantic import Field

from openhands.sdk import Action, Observation, TextContent, ImageContent
from openhands.sdk.tool import ToolDefinition, ToolExecutor
from openhands.sdk.logger import get_logger

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
        self.npcs = {}
        self.conversation_histories = {}
        self.client = None
        self.default_npc_model = None
        self.config_loaded = False
        self.problem_statement = None
        
        logger.info(f"NPCExecutor initialized (config will be loaded on first use from {config_file})")
    
    def _load_config(self):
        """Load NPC configuration from file."""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"NPC config file not found: {self.config_file}")
                return False
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Extract API configuration
            npc_api_key = config.get('npc_api_key')
            npc_base_url = config.get('npc_base_url')
            self.default_npc_model = config.get('default_npc_model', 'litellm_proxy/openai/gpt-4o')
            self.problem_statement = config.get('problem_statement')
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(
                api_key=npc_api_key,
                base_url=npc_base_url
            )
            
            # Parse NPC data
            npc_profiles = config.get('npc_profiles', [])
            npc_scenarios = config.get('npc_scenarios', [])
            
            profiles_by_name = {p['name']: p['data'] for p in npc_profiles}
            scenarios_by_name = {s['name']: s['data'] for s in npc_scenarios}
            
            # Build NPC configs
            for npc_name in profiles_by_name.keys():
                profile = profiles_by_name[npc_name]
                scenario = scenarios_by_name.get(npc_name, {})
                
                self.npcs[npc_name] = {
                    'profile': profile,
                    'scenario': scenario,
                    'model': self.default_npc_model
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
        profile: Dict,
        scenario: Dict,
        history: List[Dict],
        current_message: str
    ) -> List[Dict]:
        """Build prompt for the NPC."""
        
        background = f"""You are {npc_name}, a {profile.get('age', 'N/A')}-year-old {profile.get('occupation', 'person')}.

Your Character Profile:
- Gender: {profile.get('gender', 'N/A')} ({profile.get('gender_pronoun', 'They/Them')})
- Background: {profile.get('public_info', 'N/A')}

Your Social Goal:
{scenario.get('goal', 'Engage naturally in conversation')}

Additional Context:
{scenario.get('extra_info', '')}

Strategy Hint:
{scenario.get('strategy_hint', '')}"""
        
        instruction = f"""Imagine you are {npc_name}. Stay in character and respond naturally.

Guidelines:
1. Follow your Strategy Hint closely
2. Keep responses concise (1-3 sentences)
3. If you accept something, just acknowledge it
4. You can end the conversation when your need is met

Current situation: An AI agent is messaging you."""
        
        system_message = f"{background}\n\n{instruction}"
        
        messages = [{"role": "system", "content": system_message}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    def __call__(self, action: ChatWithNPCAction) -> NPCResponseObservation:
        """Execute NPC chat action."""
        # Lazy load config on first call
        if not self.config_loaded:
            if not self._load_config():
                return NPCResponseObservation(
                    npc_name="system",
                    response="NPC system not configured. Config file not found. Please ensure the task has been set up properly."
                )
            self.config_loaded = True
        
        npc_name = action.npc_name
        message = action.message
        
        if npc_name not in self.npcs:
            available = ', '.join(self.npcs.keys()) if self.npcs else "none"
            error_msg = f"{npc_name} does not exist. Available NPCs: {available}"
            return NPCResponseObservation(npc_name="system", response=error_msg)
        
        npc = self.npcs[npc_name]
        history = self.conversation_histories[npc_name]
        
        try:
            messages = self._build_prompt(
                npc_name, npc['profile'], npc['scenario'], history, message
            )
            
            response = self.client.chat.completions.create(
                model=npc['model'],
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            reply = response.choices[0].message.content
            
            # Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            
            if len(history) > 20:
                self.conversation_histories[npc_name] = history[-20:]
            
            return NPCResponseObservation(npc_name=npc_name, response=reply)
            
        except Exception as e:
            logger.error(f"Error from NPC {npc_name}: {e}")
            return NPCResponseObservation(
                npc_name=npc_name,
                response=f"Error: {str(e)}"
            )


def create_npc_tool() -> ToolDefinition:
    """Create NPC communication tool that reads config from workspace."""
    
    executor = NPCExecutor()
    
    description = """Chat with NPCs in the environment.

Use this tool to communicate with people mentioned in the task.
Available NPCs are configured per-task."""
    
    return ToolDefinition(
        name="chat_with_npc",
        description=description,
        action_type=ChatWithNPCAction,
        observation_type=NPCResponseObservation,
        executor=executor,
    )