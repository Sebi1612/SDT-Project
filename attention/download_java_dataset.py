from datasets import load_dataset
import json

print("Downloading dataset (this may take a few minutes)...")
ds = load_dataset("Nan-Do/code-search-net-java", split="train")

print(f"Downloaded {len(ds)} entries. Saving...")
with open("exp_data/java_full.jsonl", "w") as f:
    for entry in ds:
        entry["code_file"] = entry["sha"] + "_" + entry["func_name"]
        f.write(json.dumps(entry) + "\n")

print("Done. Saved to exp_data/java_full.jsonl")