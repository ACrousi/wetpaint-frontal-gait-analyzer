import sys
import os
import argparse
import time
import torch

# Add current dir to path
sys.path.append(os.getcwd())

from src.inference import ResGCNInference

# Mock args similar to what main.py uses
class Args:
    dataset = 'coco'
    config = 'resgcn_coco_2'
    model_type = 'resgcn'
    # Minimal model args
    model_args = {
        'num_class': 5, 
        'num_point': 17, 
        'num_person': 1, 
        'graph': 'graph.coco.Graph',
        'graph_args': {'labeling_mode': 'spatial'}
    }
    gpus = [] # CPU
    work_dir = './work_dir/test'
    pretrained_path = '' # Skip weights for this speed test
    dataset_args = {'coco': {'num_class': 5}}
    use_ldl = True
    
args = Args()

print("Starting inference initialization...")
start = time.time()

try:
    inference = ResGCNInference.from_config(args)
    end = time.time()
    print(f"Initialization SUCCESS!")
    print(f"Time taken: {end - start:.4f} seconds")
    
    # Assert it was fast (should be < 2 seconds, definitely < 60s)
    if end - start < 5.0:
        print("PASS: Initialization was fast.")
    else:
        print("FAIL: Initialization was too slow (loaded data?).")
        
except Exception as e:
    print(f"Initialization FAILED: {e}")
    import traceback
    traceback.print_exc()
