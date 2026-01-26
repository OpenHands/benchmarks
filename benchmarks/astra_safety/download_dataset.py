import json
import os

import datasets


ds_path = "PurCL/astra-agent-security"
ds = datasets.load_dataset(ds_path, split="train")
out_dir = "astra-dataset"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
fout = open(os.path.join(out_dir, "dataset.jsonl"), "w")

for entry in ds:
    fout.write(json.dumps(entry) + "\n")
fout.close()
