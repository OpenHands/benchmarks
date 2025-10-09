from __future__ import annotations

import os
from typing import Optional
from datasets import load_dataset
import pandas as pd
from openhands.sdk import get_logger, LLM, Agent, Conversation, Message, TextContent

logger = get_logger(__name__)

def load_and_verify_dataset():
    """Load and verify the OpenAgentSafety dataset."""
    DATASET = "mgulavani/openagentsafety_full"
    SPLIT = "train"  # Dataset only has train split

    try:
        # Load dataset
        logger.info(f"Loading dataset {DATASET}...")
        dataset = load_dataset(DATASET, split=SPLIT)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Print basic dataset info
        logger.info(f"Successfully loaded dataset with {len(df)} samples")
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        # Print first row as sample
        if len(df) > 0:
            logger.info("\nFirst sample in dataset:")
            first_sample = df.iloc[0]
            for col in df.columns:
                logger.info(f"{col}: {first_sample[col]}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def process_instance(instance: pd.Series, agent_llm: LLM, npc_llm: Optional[LLM] = None) -> dict:
    """Process a single OpenAgentSafety instance.
    
    Args:
        instance: Dataset row containing task information
        agent_llm: Language model to use for the agent
        npc_llm: Optional language model to use for NPC responses
        
    Returns:
        dict: Evaluation results
    """
    from npc_server import NPCServer
    
    logger.info(f"Processing instance {instance['instance_id']}")
    
    # Set up NPC interaction if needed
    has_npc_interaction = instance.get('npcs', 0) > 0 and instance.get('agent_profiles') and instance.get('agent_scenarios')
    
    # Create agent with appropriate tools
    tools = []  # Add tools if needed
    agent = Agent(llm=agent_llm, tools=tools)
    
    # Initialize conversation with direct instruction or NPC interaction
    conversation = Conversation(agent=agent)
    
    if has_npc_interaction:
        # Initialize NPC server with profile and scenario
        npc_profile = instance['agent_profiles'][0]  # Get first NPC profile
        npc_scenario = next(s for s in instance['agent_scenarios'] if s['name'] == npc_profile['name'])
        npc = NPCServer(npc_profile, npc_scenario, llm=npc_llm)
        
        # Start conversation with NPC's initial message
        npc_text = npc.get_initial_message(instance['problem_statement'])
        initial_message = Message(
            role="user",  # NPC messages are treated as user messages to the agent
            content=[TextContent(text=npc_text)],
            metadata={
                "source": "npc", 
                "npc_name": npc_profile['name'],
                "message_type": "initial"
            },
            provider_message_id=None,
            created_at=None,
            annotations=[],
            function_calls=None,
            function_responses=None,
            tool_calls=[],
            reasoning_content=None
        )
    else:
        # Direct task without NPC interaction
        initial_message = Message(
            role="user",
            content=[TextContent(text=instance['problem_statement'])],
            metadata={
                "source": "direct",
                "message_type": "task"
            },
            provider_message_id=None,
            created_at=None,
            annotations=[],
            function_calls=None,
            function_responses=None,
            tool_calls=[],
            reasoning_content=None
        )
    
    # Start the conversation
    conversation.send_message(initial_message)
    conversation.run()
    
    # For NPC tasks, continue the interaction
    if has_npc_interaction:
        agent_response = None
        # Get agent's first response
        for event in conversation.state.events:
            if hasattr(event, 'message') and event.message.role == 'assistant':
                agent_response = event.message.content[0].text
                break
                
        if agent_response:
            # Have NPC respond and continue conversation
            npc_text = npc.get_response_message(agent_response)
            npc_reply = Message(
                role="user",
                content=[TextContent(text=npc_text)],
                metadata={"source": "npc", "npc_name": npc_profile['name']},
                provider_message_id=None,
                created_at=None,
                annotations=[],
                function_calls=None,
                function_responses=None
            )
            conversation.send_message(npc_reply)
            conversation.run()
            
            # Optional third turn if agent shows resistance
            agent_response_2 = None
            for event in conversation.state.events:
                if hasattr(event, 'message') and event.message.role == 'assistant':
                    agent_response_2 = event.message.content[0].text
                    # If agent resists, give NPC one more chance
                    if agent_response_2 and any(word in agent_response_2.lower() for word in ['cannot', 'sorry', 'no', 'won\'t', 'against']):
                        npc_text = npc.get_response_message(agent_response_2)
                        npc_reply = Message(role="user", content=[TextContent(text=npc_text)])
                        conversation.send_message(npc_reply)
                        conversation.run()
                    break
    
    # Get conversation history
    history = list(conversation.state.events)
    
    result = {
        "instance_id": instance['instance_id'],
        "history": history,
        "status": "completed",
        "has_npc": has_npc_interaction
    }
    
    # Include NPC details if applicable
    if has_npc_interaction:
        result.update({
            "npc_profile": npc_profile,
            "npc_scenario": npc_scenario
        })
    
    return result

if __name__ == "__main__":
    # For local testing
    df = load_and_verify_dataset()
    
    # Create test LLM instance for agent
    agent_api_key = os.getenv("LITELLM_API_KEY")
    agent_base_url = os.getenv("LITELLM_BASE_URL")
    agent_model = os.getenv("LITELLM_MODEL")
    
    # NPC settings - fallback to agent settings if not specified
    npc_api_key = os.getenv("NPC_API_KEY", agent_api_key)
    npc_base_url = os.getenv("NPC_BASE_URL", agent_base_url)
    npc_model = os.getenv("NPC_MODEL", "litellm_proxy/neulab/gpt-4o-2024-08-06")  # Default to GPT-4 for NPCs
    
    if not all([agent_api_key, agent_base_url, agent_model]):
        raise ValueError("One or more required environment variables (LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL) are not set")
        
    # Create NPC LLM instance with consistent config
    npc_llm = LLM(
        model=npc_model,
        api_key=npc_api_key,
        base_url=npc_base_url,
        openrouter_site_url="https://docs.all-hands.dev/",
        openrouter_app_name="OpenHands",
        temperature=0.7  # Higher temperature for more dynamic NPC responses
    )
    
    # Create agent LLM instance with specific config that works with litellm
    agent_llm = LLM(
        model=agent_model,
        api_key=agent_api_key,
        base_url=agent_base_url,
        openrouter_site_url="https://docs.all-hands.dev/",
        openrouter_app_name="OpenHands",
        temperature=0,  # Low temperature for consistent agent responses
    )
    
    # Test with first instance
    test_instance = df.iloc[0]
    logger.info(f"\nTesting instance processing for {test_instance['instance_id']}")
    
    try:
        # Pass both LLM instances to process_instance
        result = process_instance(test_instance, agent_llm, npc_llm)
        logger.info(f"Processing completed: {result['status']}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")