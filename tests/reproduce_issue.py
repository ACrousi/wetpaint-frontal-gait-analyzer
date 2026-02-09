
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import ConfigManager
from src.core.workflows.prediction_workflow import PredictionWorkflow

def test_prediction_workflow_initialization():
    print("Initializing ConfigManager...")
    config_manager = ConfigManager("./config/config.yaml")
    config = config_manager.config
    workspace_root = config_manager.workspace_root
    
    print(f"Workspace Root: {workspace_root}")

    # Test 1: Initialization WITHOUT workspace_root (Should Fail)
    print("\nTest 1: Initializing PredictionWorkflow WITHOUT workspace_root...")
    try:
        workflow = PredictionWorkflow(config)
        print("FAIL: PredictionWorkflow initialized without workspace_root (Unexpected success)")
    except ValueError as e:
        print(f"PASS: Caught expected ValueError: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected exception: {type(e).__name__}: {e}")

    # Test 2: Initialization WITH workspace_root (Should Succeed)
    print("\nTest 2: Initializing PredictionWorkflow WITH workspace_root...")
    try:
        workflow = PredictionWorkflow(config, workspace_root=workspace_root)
        print("PASS: PredictionWorkflow initialized successfully with workspace_root")
    except Exception as e:
        print(f"FAIL: Initialization with workspace_root failed: {e}")

if __name__ == "__main__":
    test_prediction_workflow_initialization()
