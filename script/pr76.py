"""Evaluate the uncalibrated PR76 baseline with k_ij = 0.

This benchmark uses the same PR76 EOS, project pure-component properties, and
bubble-point solver as the calibrated PR-EOS baseline, but fixes k_ij to zero.
No experimental VLE data are used to calibrate this baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
from script.data import DATA_DIR
OUTPUTS_DIR = ROOT / "outputs"

R = 83.144  # bar cm^3 mol^-1 K^-1

# Project PR-EOS constants. Pressures are in bar.
N2 = {"Tc": 126.19, "Pc": 33.978, "omega": 0.0372}
ALKANES = {'C5N2': {'n_c': 5}, 'C6N2': {'n_c': 6}, 'C7N2': {'n_c': 7}, 'C9N2': {'n_c': 9}, 'C10N2': {'n_c': 10}, 'C12N2': {'n_c': 12}}
PROPS_FILE = DATA_DIR / "properties.csv"
Kij = 0.0


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = out.columns.astype(str).str.replace("\xa0", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def get_system_props(system: str) -> dict[str, float]:
    props = normalize_columns(pd.read_csv(PROPS_FILE))
    chemical = system.replace("N2", "")
    row = props.loc[props["Chemical"] == chemical]
    if row.empty:
        raise KeyError(f"Missing pure-component properties for {chemical}")
    row = row.iloc[0]
    return {
        "n_c": ALKANES[system]["n_c"],
        "Tc": float(row["Tc (K)"]),
        "Pc": float(row["Pc (bar)"]),
        "omega": float(row["Acentric Factor (CoolProp)"]),
    }


def pr76_pure_params(T: float, Tc: float, Pc: float, omega: float) -> tuple[float, float]:
    tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    a_tc = 0.45724 * R**2 * Tc**2 / Pc
    a_t = a_tc * (1.0 + kappa * (1.0 - math.sqrt(tr))) ** 2
    b = 0.07780 * R * Tc / Pc
    return a_t, b


def mixture_params(z_n2: float, z_alkane: float, a_n2: float, b_n2: float, a_alkane: float, b_alkane: float, kij: float) -> tuple[float, float]:
    bmix = z_n2 * b_n2 + z_alkane * b_alkane
    cross = (1.0 - kij) * math.sqrt(a_n2 * a_alkane)
    amix = z_n2**2 * a_n2 + z_alkane**2 * a_alkane + 2.0 * z_n2 * z_alkane * cross
    return amix, bmix


def z_roots(P: float, T: float, amix: float, bmix: float) -> tuple[float, float] | None:
    B = bmix * P / (R * T)
    A = amix * P / (R * T) ** 2
    if not np.isfinite(A) or not np.isfinite(B):
        return None
    B = max(B, 1e-16)
    roots = np.roots([1.0, B - 1.0, A - 3.0 * B**2 - 2.0 * B, -A * B + B**2 + B**3])
    valid = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-10 and r.real > B + 1e-12)
    if not valid:
        return None
    return valid[0], valid[-1]


def fugacity_coefficients(P: float, T: float, amix: float, bmix: float, Z: float, z_n2: float, z_alkane: float, a_n2: float, b_n2: float, a_alkane: float, b_alkane: float, kij: float) -> tuple[float, float]:
    B = bmix * P / (R * T)
    A = amix * P / (R * T) ** 2
    if not np.isfinite(Z) or Z <= B or B <= 0.0:
        return np.nan, np.nan
    sq2 = math.sqrt(2.0)
    log_term = math.log((Z + (1.0 + sq2) * B) / (Z + (1.0 - sq2) * B))
    common = A / (2.0 * sq2 * B)
    cross = (1.0 - kij) * math.sqrt(a_n2 * a_alkane)
    sum_n2 = 2.0 * (z_n2 * a_n2 + z_alkane * cross)
    sum_alkane = 2.0 * (z_n2 * cross + z_alkane * a_alkane)
    common_log = math.log(Z - B)
    phi_n2 = math.exp((b_n2 / bmix) * (Z - 1.0) - common_log - common * ((sum_n2 / amix) - (b_n2 / bmix)) * log_term)
    phi_alkane = math.exp((b_alkane / bmix) * (Z - 1.0) - common_log - common * ((sum_alkane / amix) - (b_alkane / bmix)) * log_term)
    return phi_n2, phi_alkane


def wilson_k(T: float, P: float, Pc: float, Tc: float, omega: float) -> float:
    return (Pc / P) * math.exp(5.373 * (1.0 + omega) * (1.0 - Tc / T))


def bubble_objective(P: float, T: float, x_n2: float, props: dict[str, float], direction: float, y_seed: float | None) -> dict[str, float] | None:
    omega = props["omega"]
    kij = Kij
    x_alkane = 1.0 - x_n2
    a_n2, b_n2 = pr76_pure_params(T, N2["Tc"], N2["Pc"], N2["omega"])
    a_alkane, b_alkane = pr76_pure_params(T, props["Tc"], props["Pc"], omega)
    a_liq, b_liq = mixture_params(x_n2, x_alkane, a_n2, b_n2, a_alkane, b_alkane, kij)
    roots = z_roots(P, T, a_liq, b_liq)
    if roots is None:
        return None
    z_liq, _ = roots
    phi_l_n2, phi_l_alkane = fugacity_coefficients(P, T, a_liq, b_liq, z_liq, x_n2, x_alkane, a_n2, b_n2, a_alkane, b_alkane, kij)
    if not np.isfinite(phi_l_n2) or not np.isfinite(phi_l_alkane):
        return None

    if y_seed is None:
        K_n2 = wilson_k(T, P, N2["Pc"], N2["Tc"], N2["omega"])
        K_alkane = wilson_k(T, P, props["Pc"], props["Tc"], omega)
        y_n2 = float(np.clip(x_n2 * K_n2 / (x_n2 * K_n2 + x_alkane * K_alkane), 1e-12, 1.0 - 1e-12))
    else:
        y_n2 = float(np.clip(y_seed, 1e-12, 1.0 - 1e-12))

    for _ in range(100):
        y_alkane = 1.0 - y_n2
        a_vap, b_vap = mixture_params(y_n2, y_alkane, a_n2, b_n2, a_alkane, b_alkane, kij)
        roots = z_roots(P, T, a_vap, b_vap)
        if roots is None:
            return None
        _, z_vap = roots
        phi_v_n2, phi_v_alkane = fugacity_coefficients(P, T, a_vap, b_vap, z_vap, y_n2, y_alkane, a_n2, b_n2, a_alkane, b_alkane, kij)
        if not np.isfinite(phi_v_n2) or not np.isfinite(phi_v_alkane) or phi_v_n2 <= 0.0 or phi_v_alkane <= 0.0:
            return None
        K_n2, K_alkane = phi_l_n2 / phi_v_n2, phi_l_alkane / phi_v_alkane
        s = x_n2 * K_n2 + x_alkane * K_alkane
        y_new = float(np.clip(x_n2 * K_n2 / s, 1e-12, 1.0 - 1e-12))
        if abs(y_new - y_n2) < 1e-10:
            return {"f": s - 1.0, "y_n2": y_new, "physical_score": direction * (y_new - x_n2)}
        y_n2 = y_new
    return {"f": s - 1.0, "y_n2": y_n2, "physical_score": direction * (y_n2 - x_n2)}


def objective_best(P: float, T: float, x_n2: float, props: dict[str, float], direction: float, y_hint: float | None) -> dict[str, float] | None:
    seeds = [y_hint, None, np.clip(x_n2 + 0.25 * direction, 1e-6, 1.0 - 1e-6), 0.95 if direction > 0 else 0.05, 0.75, 0.25]
    candidates = []
    used = set()
    for seed in seeds:
        key = None if seed is None else round(float(seed), 6)
        if key in used:
            continue
        used.add(key)
        result = bubble_objective(P, T, x_n2, props, direction, seed)
        if result is not None:
            candidates.append(result)
    if not candidates:
        return None
    physical = [c for c in candidates if c["physical_score"] > 1e-3]
    return max(physical or candidates, key=lambda c: (c["physical_score"], -abs(c["f"])))


def solve_bubble_pressure(T: float, x_n2: float, props: dict[str, float], direction: float, pmax_bar: float, p_previous: float | None, y_previous: float | None) -> dict[str, float] | None:
    grid = np.logspace(-1.5, math.log10(max(50.0, pmax_bar)), 96)
    if p_previous is not None and np.isfinite(p_previous):
        local = np.geomspace(max(0.03, p_previous / 2.5), min(pmax_bar, p_previous * 2.5), 32)
        grid = np.unique(np.concatenate([grid, local]))
    values, y_hint = [], y_previous
    for P in grid:
        obj = objective_best(float(P), T, x_n2, props, direction, y_hint)
        values.append(obj)
        if obj is not None:
            y_hint = obj["y_n2"]

    brackets = []
    for i, (left, right) in enumerate(zip(values[:-1], values[1:])):
        if left is None or right is None or left["physical_score"] <= 1e-2 or right["physical_score"] <= 1e-2:
            continue
        if left["f"] * right["f"] < 0.0:
            brackets.append((float(grid[i]), float(grid[i + 1]), left["y_n2"], right["y_n2"]))
    if not brackets:
        candidates = [(float(P), obj) for P, obj in zip(grid, values) if obj is not None and obj["physical_score"] > 1e-2]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item[1]["f"]))[1] | {"P_bar": min(candidates, key=lambda item: abs(item[1]["f"]))[0]}

    def score(bracket: tuple[float, float, float, float]) -> float:
        lo, hi, y_lo, y_hi = bracket
        result = 0.0
        if p_previous is not None:
            result += abs(math.log((lo + hi) / 2.0 / p_previous))
        if y_previous is not None:
            result += 8.0 * abs((y_lo + y_hi) / 2.0 - y_previous)
        return result

    lo, hi, y_lo, y_hi = min(brackets, key=score)
    f_lo = objective_best(lo, T, x_n2, props, direction, y_lo)["f"]
    y_hint = 0.5 * (y_lo + y_hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        obj = objective_best(mid, T, x_n2, props, direction, y_hint)
        if obj is None:
            hi = mid
            continue
        y_hint = obj["y_n2"]
        if abs(obj["f"]) < 1e-8 or abs(hi - lo) / mid < 1e-8:
            return obj | {"P_bar": mid}
        if f_lo * obj["f"] <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, obj["f"]
    obj = objective_best(0.5 * (lo + hi), T, x_n2, props, direction, y_hint)
    return None if obj is None else obj | {"P_bar": 0.5 * (lo + hi)}


def load_data(system: str) -> tuple[pd.DataFrame, str]:
    path = DATA_DIR / "experimental" / f"{system}.csv"
    data = normalize_columns(pd.read_csv(path))
    p_col = "P_Exp (MPa)" if "P_Exp (MPa)" in data.columns else "P_Exp (Mpa)"
    numeric = ["T (K)", "x_N2", "y_N2_Exp", p_col]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=numeric).copy()
    # Input composition columns must be nitrogen fractions.
    data = data.sort_values(["T (K)", "x_N2"]).reset_index(drop=True)
    return data, p_col


def evaluate_system(system: str) -> tuple[pd.DataFrame, dict[str, float]]:
    props = get_system_props(system)
    data, p_col = load_data(system)
    direction = 1.0 if np.median(data["y_N2_Exp"] - data["x_N2"]) >= 0.0 else -1.0
    pmax_bar = max(500.0, 1.25 * float(data[p_col].max()) * 10.0 + 10.0)
    records = []

    for T, group in data.groupby("T (K)", sort=True):
        p_previous, y_previous = None, None
        for row in group.sort_values("x_N2").itertuples(index=False):
            x_n2 = float(getattr(row, "x_N2"))
            result = solve_bubble_pressure(float(T), x_n2, props, direction, pmax_bar, p_previous, y_previous)
            if result is None:
                p_zero, y_zero = np.nan, np.nan
            else:
                p_zero, y_zero = 0.1 * result["P_bar"], result["y_n2"]
                p_previous, y_previous = result["P_bar"], y_zero
            records.append({"P_zero_MPa": p_zero, "y_zero": y_zero, "kij_zero": Kij})

    out = pd.concat([data.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    valid = out.dropna(subset=["P_zero_MPa", "y_zero"]).copy()
    p_exp = valid[p_col].to_numpy(float)
    y_exp = valid["y_N2_Exp"].to_numpy(float)
    metrics = {
        "system": system,
        "n_points": int(len(out)),
        "n_converged": int(len(valid)),
        "pressure_MARE_percent": 100.0 * float(np.mean(np.abs(valid["P_zero_MPa"].to_numpy(float) - p_exp) / np.maximum(np.abs(p_exp), 1e-6))),
        "pressure_MAE_MPa": float(np.mean(np.abs(valid["P_zero_MPa"].to_numpy(float) - p_exp))),
        "y_MARE_percent": 100.0 * float(np.mean(np.abs(valid["y_zero"].to_numpy(float) - y_exp) / np.maximum(np.abs(y_exp), 1e-3))),
        "y_MAE": float(np.mean(np.abs(valid["y_zero"].to_numpy(float) - y_exp))),
        "kij_min": float(out["kij_zero"].min()),
        "kij_max": float(out["kij_zero"].max()),
    }
    return out, metrics

