from __future__ import annotations

import argparse
import json
import math
import random
import sys
import sympy as sp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script import data as alk

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch import failed. Run this script with an environment where torch is available, "
        "install the dependencies in requirements.txt."
    ) from exc


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SYSTEMS = ["C5N2", "C6N2", "C7N2", "C9N2", "C10N2", "C12N2"]
NC_MAP = {"C5N2": 5.0, "C6N2": 6.0, "C7N2": 7.0, "C9N2": 9.0, "C10N2": 10.0, "C12N2": 12.0}
EPS = 1e-8


@dataclass
class SystemNormStats:
    tr_mean: float
    tr_std: float
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


def safe_std(v: np.ndarray) -> float:
    s = float(np.std(v))
    return s if s > 1e-12 else 1.0


def make_run_dir(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
    out = OUTPUTS_DIR / f"{stamp}_{suffix}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive(raw: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    return torch.nn.functional.softplus(raw) + floor


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -20.0, 20.0))


def load_target_dataframe(target: str) -> tuple[pd.DataFrame, dict[str, SystemNormStats], str]:
    records: list[pd.DataFrame] = []
    stats: dict[str, SystemNormStats] = {}
    for system in SYSTEMS:
        df, p_col = alk.load_preos_predictions(system)
        nc = NC_MAP[system]
        if target == "dP":
            y_raw = (df[p_col] - df["P_PR_MPa"]).to_numpy(dtype=float)
        elif target == "dy":
            y_raw = (df["y_N2_Exp"] - df["y_PR"]).to_numpy(dtype=float)
        else:
            raise ValueError(target)

        tr = df["Tr"].to_numpy(dtype=float)
        x = df["x_N2"].to_numpy(dtype=float)
        st = SystemNormStats(
            tr_mean=float(np.mean(tr)),
            tr_std=safe_std(tr),
            x_mean=float(np.mean(x)),
            x_std=safe_std(x),
            y_mean=float(np.mean(y_raw)),
            y_std=safe_std(y_raw),
        )
        stats[system] = st

        out = df.copy()
        out["system"] = system
        out["n_c"] = nc
        out["P_exp_common"] = df[p_col].to_numpy(dtype=float)
        out["tr_hat"] = (tr - st.tr_mean) / st.tr_std
        out["x_hat"] = (x - st.x_mean) / st.x_std
        out["target_raw"] = y_raw
        out["target_hat"] = (y_raw - st.y_mean) / st.y_std
        out["pressure_col_name"] = p_col
        records.append(out)
    full = pd.concat(records, ignore_index=True)
    return full, stats, p_col


class CoeffNet(nn.Module):
    def __init__(self, n_basis: int, hidden: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_basis),
        )

    def forward(self, nc: torch.Tensor) -> torch.Tensor:
        return self.net(nc)


class DpBasisLibrary(nn.Module):
    def __init__(self):
        super().__init__()
        # B1: C5-style linear-exponential competition
        self.b1 = nn.Parameter(torch.tensor([0.80, 1.67, 0.15], dtype=torch.float32))
        # B2: C7-style temperature-composition interaction
        self.b2 = nn.Parameter(torch.tensor([-0.26, 1.83, 0.51], dtype=torch.float32))
        # B3: C9-style shifted bilinear interaction
        self.b3 = nn.Parameter(torch.tensor([-0.57, 1.18, 0.90], dtype=torch.float32))
        # B4: C10-style composite exponential response
        self.b4 = nn.Parameter(torch.tensor([0.32, -0.43, 0.66, -0.09, 0.64, 0.30], dtype=torch.float32))
        # B5: C12-style inverse temperature scaling
        self.b5_raw = nn.Parameter(torch.tensor([1.54, 1.60], dtype=torch.float32))

    @property
    def n_basis(self) -> int:
        return 5

    def forward(self, tr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a1, a2, a3 = self.b1
        f1 = a1 * (x + a2 - safe_exp(a3 * tr + x))

        b1, b2, b3 = self.b2
        f2 = b1 * (tr + x.square()) * (x + b2) + b3

        c1, c2, c3 = self.b3
        f3 = c1 * (tr + x - c2) * (x + c3)

        d1, d2, d3, d4, d5, d6 = self.b4
        f4 = (d1 * tr + d2 * x) * (safe_exp(x + d3 * (tr + d4 * x.square())) - d5) + d6

        e1_raw, e2_raw = self.b5_raw
        e2 = positive(e2_raw, floor=0.2)
        f5 = (x - safe_exp(x) + e1_raw) / torch.pow(e2, tr)

        return torch.stack([f1, f2, f3, f4, f5], dim=1)

    def export(self) -> dict[str, list[float]]:
        e1_raw, e2_raw = self.b5_raw.detach().cpu().numpy().tolist()
        return {
            "basis_1_c5_style": self.b1.detach().cpu().numpy().tolist(),
            "basis_2_c7_style": self.b2.detach().cpu().numpy().tolist(),
            "basis_3_c9_style": self.b3.detach().cpu().numpy().tolist(),
            "basis_4_c10_style": self.b4.detach().cpu().numpy().tolist(),
            "basis_5_c12_style": [e1_raw, float(positive(self.b5_raw[1], floor=0.2).detach().cpu().item())],
        }


class DyBasisLibrary(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Parameter(torch.tensor([0.48, 0.30, 0.46], dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor([0.99, 0.66], dtype=torch.float32))
        self.b3 = nn.Parameter(torch.tensor([0.64, 0.94, -0.14], dtype=torch.float32))
        self.b4 = nn.Parameter(torch.tensor([1.30, 2.57, 0.012], dtype=torch.float32))
        self.b5 = nn.Parameter(torch.tensor([0.77, 0.049, 1.51, 0.053, -0.26, 0.026], dtype=torch.float32))
        self.b6_raw = nn.Parameter(torch.tensor([-0.46, 1.10], dtype=torch.float32))

    @property
    def n_basis(self) -> int:
        return 6

    def forward(self, tr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a1, a2, a3 = self.b1
        f1 = -x * (a1 + safe_exp(tr) * (a2 * x + a3))

        b1, b2 = self.b2
        f2 = b1 - x - safe_exp(b2 * tr * x)

        c1, c2, c3 = self.b3
        f3 = (c1 * tr - c2) * (x + c3 * tr.square())

        d1, d2, d3 = self.b4
        denom = tr - d2
        denom = torch.where(torch.abs(denom) < 5e-2, denom.sign() * 5e-2 + (denom == 0).float() * 5e-2, denom)
        f4 = (tr + d1) * (x / denom + d3)

        e1, e2, e3, e4, e5, e6 = self.b5
        f5 = (e1 - e2 * (x - safe_exp(x)) * safe_exp(e3 * tr)) * (e4 + e5 * x * safe_exp(tr)) - e6

        g1, g2_raw = self.b6_raw
        g2 = positive(g2_raw, floor=0.2)
        f6 = g1 * x * torch.pow(safe_exp(tr), torch.pow(g2, x))

        return torch.stack([f1, f2, f3, f4, f5, f6], dim=1)

    def export(self) -> dict[str, list[float]]:
        vals = self.b6_raw.detach().cpu().numpy().tolist()
        return {
            "basis_1_c5_style": self.b1.detach().cpu().numpy().tolist(),
            "basis_2_c6_style": self.b2.detach().cpu().numpy().tolist(),
            "basis_3_c7_style": self.b3.detach().cpu().numpy().tolist(),
            "basis_4_c9_style": self.b4.detach().cpu().numpy().tolist(),
            "basis_5_c10_style": self.b5.detach().cpu().numpy().tolist(),
            "basis_6_c12_style": [vals[0], float(positive(self.b6_raw[1], floor=0.2).detach().cpu().item())],
        }


class MSRInverseModel(nn.Module):
    def __init__(self, basis_library: nn.Module, hidden: int = 12):
        super().__init__()
        self.library = basis_library
        self.coeff_net = CoeffNet(basis_library.n_basis, hidden=hidden)

    def forward(self, tr: torch.Tensor, x: torch.Tensor, nc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = self.library(tr, x)
        alpha = self.coeff_net(nc)
        pred = torch.sum(alpha * basis, dim=1, keepdim=True)
        return pred.squeeze(1), alpha, basis



def split_by_system(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(df), dtype=bool)
    for system in SYSTEMS:
        idx = np.where(df["system"].to_numpy() == system)[0]
        n_test = max(1, int(round(test_fraction * len(idx))))
        chosen = rng.choice(idx, size=n_test, replace=False)
        mask[chosen] = True
    return pd.Series(mask, index=df.index)


def tensorize(df: pd.DataFrame, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tr": torch.tensor(df["tr_hat"].to_numpy(dtype=np.float32), device=device),
        "x": torch.tensor(df["x_hat"].to_numpy(dtype=np.float32), device=device),
        "nc": torch.tensor(df["n_c"].to_numpy(dtype=np.float32), device=device).unsqueeze(1),
        "nc_norm": None,
        "y": torch.tensor(df["target_hat"].to_numpy(dtype=np.float32), device=device),
    }


def normalize_nc(train_nc: np.ndarray, all_nc: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(train_nc))
    std = safe_std(train_nc)
    return (all_nc - mean) / std, mean, std


def train_model(df: pd.DataFrame, target: str, epochs: int, lr: float, weight_decay: float, hidden: int, seed: int, test_fraction: float) -> tuple[MSRInverseModel, pd.DataFrame, dict[str, float], dict[str, dict]]:
    device = torch.device("cpu")
    test_mask = split_by_system(df, test_fraction=test_fraction, seed=seed)
    train_df = df.loc[~test_mask].copy().reset_index(drop=True)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    work_df = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)

    nc_all = work_df["n_c"].to_numpy(dtype=float)
    nc_norm_all, nc_mean, nc_std = normalize_nc(train_df["n_c"].to_numpy(dtype=float), nc_all)
    work_df["nc_norm"] = nc_norm_all
    train_df = work_df[work_df["split"] == "train"].copy().reset_index(drop=True)
    test_df = work_df[work_df["split"] == "test"].copy().reset_index(drop=True)

    if target == "dP":
        library = DpBasisLibrary()
    elif target == "dy":
        library = DyBasisLibrary()
    else:
        raise ValueError(target)
    model = MSRInverseModel(library, hidden=hidden).to(device)

    train_tensors = tensorize(train_df, device)
    test_tensors = tensorize(test_df, device)
    train_tensors["nc"] = torch.tensor(train_df["nc_norm"].to_numpy(dtype=np.float32), device=device).unsqueeze(1)
    test_tensors["nc"] = torch.tensor(test_df["nc_norm"].to_numpy(dtype=np.float32), device=device).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_test = math.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train, alpha_train, _ = model(train_tensors["tr"], train_tensors["x"], train_tensors["nc"])
        mse = torch.mean((pred_train - train_tensors["y"]) ** 2)
        alpha_penalty = 1e-4 * torch.mean(alpha_train ** 2)
        loss = mse + alpha_penalty
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_test, _, _ = model(test_tensors["tr"], test_tensors["x"], test_tensors["nc"])
            test_mse = torch.mean((pred_test - test_tensors["y"]) ** 2).item()
        history.append({"epoch": epoch, "train_mse": float(mse.item()), "test_mse": float(test_mse)})
        if test_mse < best_test:
            best_test = test_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    pred_frames = []
    for split_df in [train_df, test_df]:
        ts = tensorize(split_df, device)
        ts["nc"] = torch.tensor(split_df["nc_norm"].to_numpy(dtype=np.float32), device=device).unsqueeze(1)
        with torch.no_grad():
            pred_hat, alpha, basis = model(ts["tr"], ts["x"], ts["nc"])
        tmp = split_df.copy()
        tmp["pred_hat"] = pred_hat.cpu().numpy()
        for j in range(alpha.shape[1]):
            tmp[f"alpha_{j+1}"] = alpha[:, j].cpu().numpy()
        for j in range(basis.shape[1]):
            tmp[f"basis_{j+1}"] = basis[:, j].cpu().numpy()
        pred_frames.append(tmp)
    pred_df = pd.concat(pred_frames, ignore_index=True)

    meta = {
        "nc_mean": nc_mean,
        "nc_std": nc_std,
        "best_test_mse": best_test,
        "final_train_mse": history[-1]["train_mse"],
        "epochs": epochs,
    }
    aux = {
        "history": history,
        "basis_parameters": model.library.export(),
    }
    return model, pred_df, meta, aux


def denormalize_predictions(df: pd.DataFrame, stats: dict[str, SystemNormStats], target: str) -> pd.DataFrame:
    out = df.copy()
    pred_raw = np.empty(len(out), dtype=float)
    for system, st in stats.items():
        mask = out["system"] == system
        pred_raw[mask] = st.y_mean + st.y_std * out.loc[mask, "pred_hat"].to_numpy(dtype=float)
    out["pred_raw"] = pred_raw
    if target == "dP":
        out["P_pred_nn"] = np.clip(out["P_PR_MPa"].to_numpy(dtype=float) + pred_raw, 0.0, None)
    else:
        out["y_pred_nn"] = np.clip(out["y_PR"].to_numpy(dtype=float) + pred_raw, 0.0, 1.0)
    return out


def eval_physical_metrics(df: pd.DataFrame, target: str) -> dict[str, float]:
    if target == "dP":
        obs = df["P_exp_common"].to_numpy(dtype=float)
        base = df["P_PR_MPa"].to_numpy(dtype=float)
        pred = df["P_pred_nn"].to_numpy(dtype=float)
        eps = 1e-6
    else:
        obs = df["y_N2_Exp"].to_numpy(dtype=float)
        base = df["y_PR"].to_numpy(dtype=float)
        pred = df["y_pred_nn"].to_numpy(dtype=float)
        eps = 1e-3
    return {
        "baseline_abs_relative_error": float(np.mean(np.abs(base - obs) / np.maximum(np.abs(obs), eps))),
        "nn_abs_relative_error": float(np.mean(np.abs(pred - obs) / np.maximum(np.abs(obs), eps))),
        "nn_mae": float(np.mean(np.abs(pred - obs))),
    }


def merge_dp_dy(dp_df: pd.DataFrame, dy_df: pd.DataFrame) -> pd.DataFrame:
    left = dp_df.copy()
    right = dy_df[["system", "T (K)", "x_N2", "Tr", "split", "y_pred_nn", "pred_raw"]].copy().rename(columns={"pred_raw": "delta_y_pred_raw"})
    left = left.rename(columns={"pred_raw": "delta_p_pred_raw"})
    merged = left.merge(right, on=["system", "T (K)", "x_N2", "Tr"], how="inner")
    return merged

