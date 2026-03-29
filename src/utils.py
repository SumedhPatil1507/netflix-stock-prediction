import os
import json

OUTPUT_DIR = "outputs"

def create_output_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_metrics(metrics):
    path = os.path.join(OUTPUT_DIR, "metrics.json")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)