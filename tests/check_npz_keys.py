import numpy as np
import os

files = [
    r"vendor\ResGCNv1\visualization\extraction_resgcn_coco.npz",
    r"vendor\ResGCNv1\visualization\extraction_resgcn_coco_eval.npz"
]

for f in files:
    print(f"Checking {f}...")
    if not os.path.exists(f):
        print(f"  FILE NOT FOUND: {f}")
        continue
    try:
        data = np.load(f, allow_pickle=True)
        print(f"  Keys: {list(data.keys())}")
        for k in data.keys():
            try:
                print(f"    {k}: shape={data[k].shape}, dtype={data[k].dtype}")
                if "label" in k or "class" in k:
                    print(f"      min={np.min(data[k])}, max={np.max(data[k])}")
            except:
                print(f"    {k}: could not get shape/details")
    except Exception as e:
        print(f"  Error loading {f}: {e}")
