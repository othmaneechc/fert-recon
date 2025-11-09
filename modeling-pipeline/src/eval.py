import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.io import load_config
from src.utils.metrics import rmse, mae, r2

# Placeholder for extra evals (spatial breakdowns, year-wise)
# For now, training prints test metrics; this script can be extended as needed.
if __name__ == "__main__":
    print(json.dumps({"note": "Evaluation is performed in train.py after early stopping."}, indent=2))
