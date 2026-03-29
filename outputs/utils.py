import json
import os

OUTPUT_DIR = "outputs"


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------
def create_output_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------
def save_metrics(metrics, filename="metrics.json"):
    path = os.path.join(OUTPUT_DIR, filename)
    
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Metrics saved at {path}")