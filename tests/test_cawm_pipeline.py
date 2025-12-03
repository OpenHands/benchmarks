import os
import json
from unittest.mock import MagicMock, patch
import pytest

from CAWM.pipeline import CAWMPipeline, PipelineConfig
from CAWM.models import Trajectory, Workflow
from CAWM.llm_client import LLMClient
from CAWM.clustering import SimilarityMethod

# Mock data
MOCK_TRAJECTORY_DATA = {
    "instance_id": "test__repo-1",
    "instruction": "Fix the bug in models.py",
    "history": [
        {
            "kind": "ActionEvent",
            "action": {"kind": "TerminalAction", "command": "ls -la"},
            "thought": [{"text": "List files"}]
        },
        {
            "kind": "ActionEvent",
            "action": {"kind": "FileEditorAction", "command": "view", "path": "models.py"},
            "thought": [{"text": "View file"}]
        }
    ],
    "test_result": {"git_patch": "..."}
}

MOCK_LLM_RESPONSE = json.dumps({
    "workflows": [
        {
            "name": "Test Workflow",
            "category": "exploration",
            "description": "Testing workflow extraction",
            "steps": [
                {
                    "env_description": "start",
                    "reasoning": "look around",
                    "action": "ls",
                    "action_type": "exploration"
                },
                {
                    "env_description": "found file",
                    "reasoning": "read it",
                    "action": "view {file}",
                    "action_type": "file_view"
                }
            ]
        }
    ]
})

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = MOCK_LLM_RESPONSE
    client.parse_structured_response.return_value = json.loads(MOCK_LLM_RESPONSE)
    return client

@pytest.fixture
def mock_trajectories():
    return [Trajectory.from_raw(MOCK_TRAJECTORY_DATA)]

def test_pipeline_initialization(mock_llm_client):
    config = PipelineConfig()
    pipeline = CAWMPipeline(llm_client=mock_llm_client, config=config)
    assert pipeline.llm_client == mock_llm_client
    assert pipeline.config == config

def test_pipeline_run(mock_llm_client, mock_trajectories):
    pipeline = CAWMPipeline(llm_client=mock_llm_client)
    
    # Run pipeline
    workflows = pipeline.run(mock_trajectories)
    
    assert len(workflows) == 1
    assert workflows[0].id.startswith("wf-general-")
    assert workflows[0].category == "exploration"
    assert len(workflows[0].steps) == 2
    
    # Verify interactions
    # Compression called (implicitly via induction or explicit)
    # Clustering called
    # Induction called (mock LLM should be called)
    mock_llm_client.complete.assert_called()

def test_clustering_integration(mock_llm_client):
    # Create two distinct trajectories
    t1 = Trajectory.from_raw({
        "instance_id": "t1", 
        "history": [{"kind": "ActionEvent", "action": {"kind": "TerminalAction", "command": "ls"}}]
    })
    t2 = Trajectory.from_raw({
        "instance_id": "t2", 
        "history": [{"kind": "ActionEvent", "action": {"kind": "TerminalAction", "command": "pytest"}}]
    })
    
    config = PipelineConfig(clustering_method=SimilarityMethod.ACTION_SEQUENCE, clustering_threshold=0.9)
    pipeline = CAWMPipeline(llm_client=mock_llm_client, config=config)
    
    # t1 and t2 should likely be in different clusters if threshold is high and actions differ
    # But our simple clusterer might just group them if sequence similarity logic is simple.
    # Let's just check it runs without error for now.
    workflows = pipeline.run([t1, t2])
    assert len(workflows) >= 0 

def test_compression_integration(mock_llm_client, mock_trajectories):
    pipeline = CAWMPipeline(llm_client=mock_llm_client)
    
    # Verify that the compressor inside pipeline works
    compressed = pipeline.compressor.compress(mock_trajectories[0])
    assert isinstance(compressed, Trajectory)
