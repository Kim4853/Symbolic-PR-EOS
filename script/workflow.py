"""Notebook API and command-line entry points. Run from the repository root."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from script.data import PROJECT_ROOT, DATA_DIR, SYSTEMS, load_preos_predictions, save_json, validate_system


def make_output(name):
    folder = PROJECT_ROOT / "outputs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{name}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def environment():
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "matplotlib", "sympy", "torch", "pysr"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def discover(system, seed=1, iterations=None):
    """Search system-specific residual expressions; does not redefine shared bases."""
    from pysr import PySRRegressor
    from script.data import normalize_training_data
    from script.discovery_weights import expand_dp_training, expand_dy_training
    validate_system(system)
    config = json.loads((PROJECT_ROOT / "configs/search_settings.json").read_text())[system]
    if iterations is not None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        config["niterations"] = iterations
    folder = make_output(f"discovery_{system}")
    baseline, p_col = load_preos_predictions(system)
    summary = []
    for target, expand in [("dP", expand_dp_training), ("dy", expand_dy_training)]:
        x, y = expand(system, baseline, p_col) if target == "dP" else expand(system, baseline)
        x, y, stats = normalize_training_data(x, y)
        options = dict(config, populations=31, parsimony=0.0, random_state=seed,
                       optimizer_algorithm="BFGS", optimizer_iterations=8,
                       optimizer_nrestarts=2, should_optimize_constants=True,
                       early_stop_condition=None,
                       elementwise_loss="loss(x, y) = (x - y)^2",
                       progress=False, output_directory=str(folder), run_id=target)
        model = PySRRegressor(**options)
        model.fit(x, y)
        candidates = model.equations_.sort_values("complexity")
        candidates.to_csv(folder / f"{target}_candidates.csv", index=False)
        chosen = candidates.iloc[-1]
        summary.append({"target": target, "expression_normalized": str(chosen["equation"]),
                        "complexity": int(chosen["complexity"]), "normalized_training_mse": float(chosen["loss"])})
        save_json(folder / f"{target}_normalization.json", stats)
    save_json(folder / "settings.json", {"system": system, "seed": seed, "search": config,
                                         "populations": 31, "environment": environment(),
                                         "selection": "largest complexity retained on final candidate front"})
    save_json(folder / "selected_expressions.json", summary)
    return folder


def train_shared(epochs=2500, lr=5e-3, seed=42, hidden=12, weight_decay=1e-6, test_fraction=0.2):
    """Jointly fit the fixed shared basis and carbon-number coefficient networks."""
    import torch
    from script import shared_model as model
    if epochs < 1 or lr <= 0 or not 0 < test_fraction < 1:
        raise ValueError("Require epochs >= 1, lr > 0 and 0 < test_fraction < 1")
    folder = make_output("shared_model")
    model.set_seed(seed)
    physical, metadata = {}, {}
    for target, split_seed in [("dP", seed), ("dy", seed + 1)]:
        raw, stats, _ = model.load_target_dataframe(target)
        network, pred, meta, auxiliary = model.train_model(
            raw, target=target, epochs=epochs, lr=lr, weight_decay=weight_decay,
            hidden=hidden, seed=split_seed, test_fraction=test_fraction)
        physical[target] = model.denormalize_predictions(pred, stats, target)
        torch.save(network.state_dict(), folder / f"{target}_state_dict.pt")
        metadata[target] = {**meta, **auxiliary}
        save_json(folder / f"{target}_training.json", metadata[target])
        save_json(folder / f"{target}_normalization.json", {key: asdict(value) for key, value in stats.items()})
        coefficients = pred.groupby("system").first().reset_index()
        cols = [c for c in pred if c.startswith("alpha_")]
        coefficients[["system", "n_c"] + cols].to_csv(folder / f"{target}_coefficients.csv", index=False)
        polynomial = {c: np.polynomial.polynomial.polyfit(
            coefficients.n_c.to_numpy(float), coefficients[c].to_numpy(float), 5).tolist() for c in cols}
        save_json(folder / f"{target}_polynomials.json", {
            "coefficient_order": "ascending powers of carbon number", "degree": 5, "coefficients": polynomial})
    merged = model.merge_dp_dy(physical["dP"], physical["dy"])
    assert len(merged) == sum(len(load_preos_predictions(s)[0]) for s in SYSTEMS)
    merged.to_csv(folder / "predictions.csv", index=False)
    save_json(folder / "settings.json", {"epochs": epochs, "lr": lr, "seed": seed, "hidden": hidden,
        "weight_decay": weight_decay, "coefficient_penalty": 1e-4, "holdout_fraction": test_fraction,
        "selection": "minimum monitored holdout MSE; not an independent test estimate",
        "normalization": "system-specific statistics over all cached states before splitting",
        "input_kind": "fictional demo" if (DATA_DIR / "SYNTHETIC_DEMO.json").exists() else "user-provided",
        "systems": SYSTEMS, "environment": environment()})
    return folder


def recompute_baselines(systems=None, kind="generalized"):
    """Write a new baseline run without overwriting distributed caches."""
    from script import pr76, yang2022
    if kind not in ["zero", "calibrated", "generalized"]:
        raise ValueError(kind)
    folder = make_output(f"baseline_{kind}")
    bips = json.loads((PROJECT_ROOT / "configs/calibrated_bip.json").read_text())
    reports = []
    try:
        for system in systems or SYSTEMS:
            validate_system(system)
            if kind == "generalized":
                frame, report = yang2022.evaluate_system(system)
            else:
                pr76.Kij = bips[system] if kind == "calibrated" else 0.0
                frame, report = pr76.evaluate_system(system)
                if kind == "calibrated":
                    frame = frame.rename(columns={"P_zero_MPa": "P_PR_MPa", "y_zero": "y_PR", "kij_zero": "kij"})
            frame.to_csv(folder / f"{system}.csv", index=False)
            reports.append(report)
    finally:
        pr76.Kij = 0.0
    # JSON null denotes a metric that could not be calculated (no finite roots).
    reports = json.loads(pd.DataFrame(reports).to_json(orient="records"))
    save_json(folder / "solver_report.json", reports)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-shared")
    train.add_argument("--epochs", type=int, default=2500)
    train.add_argument("--lr", type=float, default=5e-3)
    train.add_argument("--seed", type=int, default=42)
    search = commands.add_parser("discover")
    search.add_argument("--system", choices=SYSTEMS, required=True)
    search.add_argument("--seed", type=int, default=1)
    search.add_argument("--iterations", type=int)
    plot = commands.add_parser("plot")
    plot.add_argument("--system", choices=SYSTEMS)
    plot.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "outputs/proposed_predictions.csv")
    plot.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/figures")
    baseline = commands.add_parser("baselines")
    baseline.add_argument("--model", choices=["zero", "calibrated", "generalized"], default="generalized")
    baseline.add_argument("--system", choices=SYSTEMS)
    args = parser.parse_args()
    if args.command == "discover":
        result = discover(args.system, seed=args.seed, iterations=args.iterations)
    elif args.command == "train-shared":
        result = train_shared(args.epochs, args.lr, args.seed)
    elif args.command == "baselines":
        result = recompute_baselines([args.system] if args.system else None, args.model)
    else:
        from script.plotting import plot_system
        result = [plot_system(s, args.predictions, args.out) for s in ([args.system] if args.system else SYSTEMS)]
    print(result)


if __name__ == "__main__":
    main()
