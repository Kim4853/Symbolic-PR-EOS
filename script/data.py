"""Dataset access shared by discovery and coefficient fitting."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SYMBOLIC_PR_EOS_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SYSTEMS = ["C5N2", "C6N2", "C7N2", "C9N2", "C10N2", "C12N2"]


def validate_system(system):
    if system not in SYSTEMS:
        raise ValueError(f"Choose one of {SYSTEMS}, got {system!r}")
    return system


def load_preos_predictions(system):
    validate_system(system)
    path = DATA_DIR / "calibrated" / f"{system}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Supply your own data as described in data/README.md, or generate the synthetic demo.")
    frame = pd.read_csv(path)
    return frame, "P_Exp (MPa)"


def get_system_props(system):
    validate_system(system)
    frame = pd.read_csv(DATA_DIR / "properties.csv")
    row = frame.loc[frame.Chemical == system.replace("N2", "")].iloc[0]
    return float(row["Tc (K)"]), float(row["Pc (bar)"]), float(row["Acentric Factor (CoolProp)"])


def normalize_training_data(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    stats = {"x_mean": x.mean(axis=0).tolist(), "x_scale": scale.tolist(),
             "y_mean": float(y.mean()), "y_scale": float(y.std()) if y.std() > 1e-12 else 1.0}
    return (x - stats["x_mean"]) / scale, (y - stats["y_mean"]) / stats["y_scale"], stats


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
