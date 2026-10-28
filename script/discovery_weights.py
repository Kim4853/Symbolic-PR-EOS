"""Historical C5 discovery weights; not applied during shared-model fitting."""
import numpy as np
import pandas as pd

def expand_dp_training(system: str, baseline_df: pd.DataFrame, pressure_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = baseline_df[["Tr", "x_N2"]].to_numpy(dtype=float)
    y = (baseline_df[pressure_col] - baseline_df["P_PR_MPa"]).to_numpy(dtype=float)
    if system != "C5N2":
        return x, y
    tr = baseline_df["Tr"].to_numpy(dtype=float)
    xn2 = baseline_df["x_N2"].to_numpy(dtype=float)
    weights = np.ones(len(baseline_df), dtype=int)
    hotspot = (tr > 0.93) & (xn2 > 0.20) & (xn2 < 0.27)
    shoulder = (tr > 0.90) & (xn2 > 0.17) & (xn2 < 0.30)
    weights[shoulder] = np.maximum(weights[shoulder], 3)
    weights[hotspot] = np.maximum(weights[hotspot], 6)
    return np.repeat(x, weights, axis=0), np.repeat(y, weights, axis=0)


def expand_dy_training(system: str, baseline_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = baseline_df[["Tr", "x_N2"]].to_numpy(dtype=float)
    y = (baseline_df["y_N2_Exp"] - baseline_df["y_PR"]).to_numpy(dtype=float)
    if system != "C5N2":
        return x, y
    y_exp = baseline_df["y_N2_Exp"].to_numpy(dtype=float)
    weights = np.clip(np.ceil(0.45 / np.maximum(y_exp, 0.08)), 1, 5).astype(int)
    return np.repeat(x, weights, axis=0), np.repeat(y, weights, axis=0)


