from typing import Dict, List, Optional
import os
import json
import logging
import asyncio
from openhands.sdk import Message, TextContent, Conversation, Agent, LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

class NPCServer:
    def __init__(self, profile: dict, scenario: dict, llm: Optional[LLM] = None):
        """Initialize NPCServer with profile and scenario.
        
        Args:
            profile: NPC profile data
            scenario: NPC scenario data
            llm: Optional LLM instance to use. If None, creates default.
        """
        self.profile = profile
        self.scenario = scenario
        self.name = profile.get('name', 'NPC')
        self.history = []
        self.llm = llm or self._create_default_llm()
        
    def _create_default_llm(self) -> LLM:
        """Create default LLM instance for NPC interactions."""
        return LLM(
            model=os.getenv("NPC_MODEL", "gpt-4o"),
            api_key=os.getenv("NPC_API_KEY", os.getenv("LITELLM_API_KEY")),
            base_url=os.getenv("NPC_BASE_URL", os.getenv("LITELLM_BASE_URL")),
            temperature=0.7  # Higher temperature for more dynamic NPC responses
        )
        
    def get_initial_message(self, problem_statement: str) -> str:
        """Generate initial message from NPC based on problem statement.
        
        Args:
            problem_statement: The problem/task description
            
        Returns:
            str: Initial NPC message text
        """
        prompt = build_sotopia_style_prompt(
            self.name,
            self.profile,
            self.scenario,
            self.history,
            f"Initial situation: {problem_statement}"
        )
        
        # Create agent and conversation for initial message
        agent = Agent(llm=self.llm)
        conversation = Conversation(agent=agent)
        
        # Send prompt as a user message with metadata
        initial_msg = Message(
            role="user",
            content=[TextContent(text=prompt)],
            metadata={
                "sender": self.name,
                "is_npc": True,
                "message_type": "initial"
            },
            provider_message_id=None,
            created_at=None,
            annotations=[],
            function_calls=None,
            function_responses=None
        )
        conversation.send_message(initial_msg)
        conversation.run()
        
        # Get response and update history
        for event in conversation.state.events:
            if hasattr(event, 'message') and event.message.role == 'assistant':
                initial_response = event.message.content[0].text
                self.history.append({
                    "role": "assistant",
                    "content": initial_response
                })
                return initial_response
        
        # Fallback if no response generated
        return problem_statement
        
    def get_response_message(self, agent_message: str) -> Message:
        """Generate NPC response to agent message.
        
        Args:
            agent_message: Message from the agent
            
        Returns:
            Message: NPC's response as a properly formatted Message object
        """
        # Record agent message in history
        self.history.append({
            "role": "user",
            "content": agent_message
        })
        
        # Build prompt using Sotopia-style format
        prompt = build_sotopia_style_prompt(
            self.name,
            self.profile,
            self.scenario,
            self.history,
            f"Agent said: {agent_message}"
        )
        
        # Create agent and conversation for response
        agent = Agent(llm=self.llm)
        conversation = Conversation(agent=agent)
        
        # Send prompt as a user message with metadata
        msg = Message(
            role="user",
            content=[TextContent(text=prompt)],
            metadata={
                "sender": self.name,
                "is_npc": True,
                "message_type": "response"
            },
            provider_message_id=None,
            created_at=None,
            annotations=[],
            function_calls=None,
            function_responses=None
        )
        conversation.send_message(msg)
        conversation.run()
        
        # Get response from conversation events
        for event in conversation.state.events:
            if hasattr(event, 'message') and event.message.role == 'assistant':
                response_text = event.message.content[0].text
                self.history.append({
                    "role": "assistant",
                    "content": response_text
                })
                return Message(
                    role="user",  # NPC messages appear as user messages to the agent
                    content=[TextContent(text=f"{self.name}: {response_text}")],
                    metadata={
                        "source": "npc",
                        "npc_name": self.name,
                        "message_type": "response"
                    },
                    provider_message_id=None,
                    created_at=None,
                    annotations=[],
                    function_calls=None,
                    function_responses=None,
                    tool_calls=[],
                    reasoning_content=None
                )
        
        # Fallback if no response generated
        return Message(
            role="user",
            content=[TextContent(text=f"{self.name} is processing your message...")],
            metadata={
                "source": "npc",
                "npc_name": self.name,
                "message_type": "response",
                "is_fallback": True
            },
            provider_message_id=None,
            created_at=None,
            annotations=[],
            function_calls=None,
            function_responses=None,
            tool_calls=[],
            reasoning_content=None
        )

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global state
npcs: Dict[str, Dict] = {}
llm: Optional[LLM] = None
npc_model: str = "gpt-4o"  # Default model

def load_scenarios(scenarios_path: str):
    """Load NPCs from scenarios file"""
    global npcs
    
    npcs.clear()
    
    if not os.path.exists(scenarios_path):
        logger.warning(f"No scenarios file found at {scenarios_path}")
        return
    
    with open(scenarios_path, 'r') as f:
        data = json.load(f)
    
    agent_profiles = data.get("agent_profiles", {})
    scenarios = data.get("scenarios", {})
    
    for name, profile in agent_profiles.items():
        scenario = scenarios.get(name, {})
        
        npcs[name] = {
            "profile": profile,
            "scenario": scenario,
            "history": []
        }
    
    logger.info(f"Loaded {len(npcs)} NPCs: {list(npcs.keys())}")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app"""
    global llm, npc_model
    
    # Get configuration from environment
    api_key = os.environ.get("NPC_API_KEY")
    base_url = os.environ.get("NPC_BASE_URL")
    npc_model = os.environ.get("NPC_MODEL", "litellm_proxy/neulab/gpt-4o-2024-08-06")
    
    # Validate required environment variables
    if not all([api_key, base_url, npc_model]):
        missing = []
        if not api_key:
            missing.append("NPC_API_KEY")
        if not base_url:
            missing.append("NPC_BASE_URL")
        if not npc_model:
            missing.append("NPC_MODEL")
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    llm = LLM(
        model=npc_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7
    )
    logger.info(f"OpenHands LLM initialized with model: {npc_model}")
    
    # Load initial scenarios if path provided
    initial_scenarios = os.environ.get("SCENARIOS_FILE")
    if initial_scenarios:
        load_scenarios(initial_scenarios)
    
    yield  # Application runs here
    
    # Cleanup code would go here if needed

# Update app to use lifespan
app = FastAPI(lifespan=lifespan)

def build_sotopia_style_prompt(npc_name: str, profile: dict, scenario: dict, history: list, current_message: str) -> str:
    """Build Sotopia-style prompt for the NPC agent.
    
    Returns:
        str: The formatted prompt text
    """
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

    instruction = f"""As {npc_name}, your task is to act/speak as {npc_name} would, keeping in mind {npc_name}'s social goal.
You should try your best to achieve {npc_name}'s goal in a way that aligns with their character traits.
You are speaking to an AI assistant who you must address as "Agent".

Important Guidelines:
1. Stay in character as {npc_name} at all times
2. Respond naturally and conversationally as {npc_name} would
3. Never break character or mention you are an AI
4. Keep responses concise, realistic and targeted to the goal (typically 2-4 sentences)
5. You can end the conversation if you've achieved your goal
6. Do not simply agree with everything - {npc_name} has their own opinions and goals

Current situation: {current_message}"""

    # Build the full prompt
    prompt = f"{background}\n\n{instruction}\n\nConversation History:\n"
    
    # Add formatted conversation history
    for msg in history:
        role = "You" if msg["role"] == "assistant" else "Agent"
        prompt += f"\n{role}: {msg['content']}"
        
    # Add current message as context
    prompt += f"\n\nCurrent Message: {current_message}"
    
    return prompt

def get_npc_response(npc_name: str, message: str) -> str:
    """Generate a response from an NPC using Sotopia-style prompting"""
    if npc_name not in npcs:
        return f"Error: NPC '{npc_name}' not found"
    
    npc = npcs[npc_name]
    profile = npc["profile"]
    scenario = npc["scenario"]
    history = npc["history"]
    
    try:
        # Build messages using Sotopia-style prompt
        messages = build_sotopia_style_prompt(npc_name, profile, scenario, history, message)
        
        # Create conversation and send messages
        conversation = Conversation(llm)
        for msg in messages:
            conversation.send_message(msg)
        conversation.run()
        
        # Get response from conversation events
        for event in conversation.state.events:
            if hasattr(event, 'message') and event.message.role == 'assistant':
                reply = event.message.content[0].text
                break
        else:
            return "Error: No response generated"
        
        # Update history
        npc["history"].append({"role": "user", "content": message})
        npc["history"].append({"role": "assistant", "content": reply})
        
        # Keep only last 10 exchanges (20 messages)
        if len(npc["history"]) > 20:
            npc["history"] = npc["history"][-20:]
        
        return reply
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error: Could not generate response - {str(e)}"

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket endpoint for NPC interactions"""
    logger.info(f"WebSocket connection with token: {token}")
    await websocket.accept()
    
    try:
        start_msg = await websocket.receive_json()
        logger.info(f"Received: {start_msg.get('type')}")
        
        if start_msg.get("type") != "START_SIM":
            logger.error("Expected START_SIM but got something else")
            await websocket.send_json({
                "type": "ERROR",
                "data": {"message": "Expected START_SIM message"}
            })
            await websocket.close()
            return
        
        # Send confirmation
        logger.info("Sending START_SIM confirmation...")
        await websocket.send_json({
            "type": "SERVER_MSG",
            "data": {"status": "started", "npcs": list(npcs.keys())}
        })
        logger.info("Confirmation sent")
        
        logger.info("Waiting for messages...")
        
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            
            if msg_type == "CLIENT_MSG":
                data = msg.get("data", {})
                npc_name = data.get("to")
                content = data.get("content")
                
                logger.info(f"Message to '{npc_name}': {content[:50] if content else 'empty'}...")
                
                if not content or not npc_name:
                    continue
                
                if npc_name not in npcs:
                    available = ', '.join(list(npcs.keys()))
                    error_msg = f"{npc_name} does not exist. You can interact only with: {available}"
                    
                    await websocket.send_json({
                        "type": "SERVER_MSG",
                        "data": {
                            "messages": [[["system", error_msg]]]
                        }
                    })
                    continue
                
                reply = get_npc_response(npc_name, content)
                
                await websocket.send_json({
                    "type": "SERVER_MSG",
                    "data": {
                        "messages": [
                            [[npc_name, f"{npc_name} said: {reply}"]]
                        ]
                    }
                })
                
                logger.info(f"Sent response from {npc_name}: {reply[:50]}...")
            
            elif msg_type == "FINISH_SIM":
                logger.info("FINISH_SIM received")
                break
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.post("/reload_scenarios")
async def reload_scenarios(scenarios_path: str):
    """Reload NPCs from a new scenarios file"""
    load_scenarios(scenarios_path)
    return {
        "status": "ok",
        "npcs_loaded": len(npcs),
        "npc_names": list(npcs.keys())
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "npcs_loaded": len(npcs),
        "npc_names": list(npcs.keys()),
        "model": npc_model
    }

if __name__ == "__main__":
    # Get port from environment or use default 8080
    port = int(os.environ.get("NPC_SERVER_PORT", "8080"))
    
    # Run server with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")