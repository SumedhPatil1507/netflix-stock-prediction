import os
import json
import logging
import csv
from datetime import datetime

OUTPUT_DIR = "outputs"
EXPERIMENT_LOG = os.path.join(OUTPUT_DIR, "experiment_log.csv")

logger = logging.getLogger(__name__)


def create_output_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def save_metrics(metrics: dict):
    path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved → {path}")


def log_experiment(params: dict, metrics: dict):
    """Append a run to the experiment log CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update(params)
    row.update(metrics)

    file_exists = os.path.exists(EXPERIMENT_LOG)
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Experiment logged → {EXPERIMENT_LOG}")


def setup_logging(level: str = "INFO"):
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("netflix")
