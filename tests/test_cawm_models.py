import os
from CAWM.models import Trajectory, ActionType, Workflow, WorkflowStep, abstract_path

def test_workflow_classes():
    step = WorkflowStep(
        env_description="test env",
        reasoning="test reasoning",
        action="ls",
        action_type="exploration"
    )
    workflow = Workflow(
        id="wf-1",
        description="test workflow",
        category="test",
        steps=[step],
        level=1
    )
    
    assert workflow.to_dict()["id"] == "wf-1"
    assert workflow.steps[0].action == "ls"

def test_helper_functions():
    path = "/workspace/django/models.py"
    abs_path = abstract_path(path, "django")
    assert abs_path == "{repo}/models.py", f"Expected {{repo}}/models.py, got {abs_path}"

def test_trajectory_loading():
    jsonl_path = "CAWM/trajectories/resolved_trajectories.jsonl"
    if os.path.exists(jsonl_path):
        trajectories = Trajectory.load_from_jsonl(jsonl_path)
        
        if len(trajectories) > 0:
            traj = trajectories[0]
            
            # Check key events
            key_events = traj.get_key_events()
            
            # Verify some action types
            found_classified = False
            for event in traj.events:
                if event.action_type != ActionType.OTHER:
                    found_classified = True
                    break
            
            # We expect at least some actions to be classified in a real trajectory
            # But if the trajectory has no recognizable actions, this might fail.
            # Based on previous run, it found THINK action.
            assert found_classified

