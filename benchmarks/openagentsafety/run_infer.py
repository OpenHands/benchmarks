from __future__ import annotations

import os
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

def process_instance(instance: pd.Series, llm: LLM) -> dict:
    """Process a single OpenAgentSafety instance.
    
    Args:
        instance: Dataset row containing task information
        llm: Language model to use for evaluation
        
    Returns:
        dict: Evaluation results
    """
    logger.info(f"Processing instance {instance['instance_id']}")
    
    # Create agent and conversation
    agent = Agent(llm=llm)
    conversation = Conversation(agent=agent)
    
    # Prepare task instruction
    task_instruction = instance['problem_statement']
    
    # Create and send message
    message = Message(
        role="user",
        content=[TextContent(text=task_instruction)]  # Pass as a list
    )
    conversation.send_message(message)
    conversation.run()
    
    # Get conversation history
    history = list(conversation.state.events)
    
    return {
        "instance_id": instance['instance_id'],
        "history": history,
        "status": "completed"
    }

if __name__ == "__main__":
    # For local testing
    df = load_and_verify_dataset()
    
    # Create test LLM instance
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL")
    model = os.getenv("LITELLM_MODEL")
    
    if not all([api_key, base_url, model]):
        raise ValueError("One or more required environment variables (LITELLM_API_KEY, LITELLM_BASE_URL, LITELLM_MODEL) are not set")
    
    llm = LLM(
        model=model,  # Add custom/ prefix for LiteLLM
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    
    # Test with first instance
    test_instance = df.iloc[0]
    logger.info(f"\nTesting instance processing for {test_instance['instance_id']}")
    
    try:
        result = process_instance(test_instance, llm)
        logger.info(f"Processing completed: {result['status']}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")