import os
import sys
import logging
import json
from typing import List

# Add project root to path (one level up from tests/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CAWM import (
    CAWMPipeline, 
    PipelineConfig, 
    LLMClient, 
    Trajectory, 
    CompressionStrategy, 
    SimilarityMethod, 
    WorkflowLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_trajectories(path: str, n: int = 3) -> List[Trajectory]:
    """Load first n trajectories from file."""
    logger.info(f"Loading top {n} trajectories from {path}...")
    trajectories = []
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                if line.strip():
                    data = json.loads(line)
                    trajectories.append(Trajectory.from_raw(data))
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)
    return trajectories

def main():
    # 1. Check API Key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No API key found! Please set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")
        sys.exit(1)

    # 2. Load Data
    # Path relative to project root
    input_path = os.path.join(os.path.dirname(__file__), "../CAWM/trajectories/resolved_trajectories.jsonl")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    trajectories = load_sample_trajectories(input_path, n=3)
    logger.info(f"Loaded {len(trajectories)} trajectories.")

    # 3. Configure Pipeline (Testing Stage 2 features)
    config = PipelineConfig(
        compression_strategy=CompressionStrategy.HIERARCHICAL_SUMMARIZATION,
        clustering_method=SimilarityMethod.ACTION_SEQUENCE,
        workflow_level=WorkflowLevel.GENERAL,
        clustering_threshold=0.3
    )

    # 4. Initialize Client
    provider = "openrouter"
    if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        provider = "anthropic"
    elif os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        provider = "openai"
        config.llm_model = "gpt-4o"

    logger.info(f"Initializing Pipeline with provider={provider}...")
    llm_client = LLMClient(provider=provider)
    
    pipeline = CAWMPipeline(llm_client=llm_client, config=config)

    # 5. Run
    logger.info("Running CAWM Pipeline...")
    workflows = pipeline.run(trajectories)

    # 6. Save Results
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "induced_workflows.json")
    
    results = {
        "meta": {
            "input_file": input_path,
            "trajectory_count": len(trajectories),
            "config": {
                "compression": config.compression_strategy.name,
                "clustering": config.clustering_method.name,
                "level": config.workflow_level.name
            }
        },
        "workflows": [wf.to_dict() for wf in workflows]
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*50)
    print(f"Successfully Induced {len(workflows)} Workflows")
    print(f"Results saved to: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
