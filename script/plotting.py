"""Temperature-panel VLE plots with trace-only numerical safeguards."""
from pathlib import Path
from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from script.data import PROJECT_ROOT, DATA_DIR, get_system_props, validate_system

STYLES = [
    ("Experiment", "#D62828", "-", 2.4),
    (r"PR-EOS ($k_{ij}=0$)", "#666666", ":", 1.7),
    (r"PR-EOS (calibrated $k_{ij}$)", "black", "-.", 1.8),
    (r"PR-EOS (generalized $k_{ij}$)", "#E07A16", "-.", 1.8),
    ("Two-level SR (proposed)", "#0000FF", "--", 2.1),
]


def smooth_branch(composition, pressure):
    """Interpolate each branch without extrapolating its sampled pressure range."""
    order = np.argsort(pressure)
    pressure, composition = np.asarray(pressure)[order], np.asarray(composition)[order]
    pressure, indices = np.unique(pressure, return_index=True)
    composition = composition[indices]
    if len(pressure) < 3:
        return composition, pressure
    grid = np.linspace(pressure[0], pressure[-1], 200)
    return PchipInterpolator(pressure, composition)(grid), grid


def reference_subset(frame, reference):
    def keyed(data):
        return data.assign(_T=data["T (K)"].round(6), _x=data.x_N2.round(8))
    return keyed(frame).merge(keyed(reference)[["_T", "_x"]].drop_duplicates(),
                              on=["_T", "_x"], how="inner", validate="one_to_one")


def trace_segments(frame, p_col, y_col, is_model=True):
    """Break at missing/trivial roots or abrupt downward pressure jumps.

    This is a plotting heuristic, not a phase-stability or criticality test.
    Removed rows are returned for an explicit plotting audit.
    """
    segments, active, excluded = [], [], []
    previous = None
    for _, row in frame.sort_values("x_N2").iterrows():
        p, y, x = row[p_col], row[y_col], row.x_N2
        reason = None
        if not np.isfinite([p, y, x]).all() or p < 0 or not (0 <= y <= 1):
            reason = "nonfinite or outside physical plotting bounds"
        elif is_model and abs(y - x) <= 1e-6:
            reason = "composition gap <= 1e-6"
        elif is_model and previous is not None and p < 0.9 * previous:
            reason = "pressure drop > 10% along increasing liquid composition"
        if reason:
            if active:
                segments.append(pd.DataFrame(active))
                active = []
            excluded.append({"T (K)": row["T (K)"], "x_N2": x, "pressure": p, "reason": reason})
        else:
            active.append(row)
            previous = p
    if active:
        segments.append(pd.DataFrame(active))
    return segments, excluded


def plot_system(system, predictions=None, output_dir=None, baseline_dirs=None):
    """Plot one system; baseline_dirs may override cached model directories."""
    validate_system(system)
    predictions = PROJECT_ROOT / "outputs/proposed_predictions.csv" if predictions is None else Path(predictions)
    output_dir = PROJECT_ROOT / "outputs/figures" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not predictions.exists():
        raise FileNotFoundError(f"Missing predictions: {predictions}. Train a model and pass its predictions.csv, or generate the synthetic demo.")
    proposed = pd.read_csv(predictions)
    proposed = proposed.loc[proposed.system == system].copy()
    if proposed.empty:
        raise ValueError(f"No {system} rows in {predictions}")
    dirs = {"zero": DATA_DIR / "baselines/zero",
            "calibrated": DATA_DIR / "calibrated",
            "generalized": DATA_DIR / "baselines/generalized"}
    dirs.update(baseline_dirs or {})
    experiment = pd.read_csv(DATA_DIR / "experimental" / f"{system}.csv")
    traces = [(experiment, "P_Exp (MPa)", "y_N2_Exp")]
    for kind, p, y in [("zero", "P_zero_MPa", "y_zero"), ("calibrated", "P_PR_MPa", "y_PR"),
                       ("generalized", "P_yang_MPa", "y_yang")]:
        baseline = pd.read_csv(Path(dirs[kind]) / f"{system}.csv")
        traces.append((reference_subset(baseline, proposed), p, y))
    traces.append((proposed, "P_pred_nn", "y_pred_nn"))
    temperatures = np.sort(experiment["T (K)"].unique())
    carbon = int(system.split("N")[0][1:])
    tc, _, _ = get_system_props(system)
    synthetic = (DATA_DIR / "SYNTHETIC_DEMO.json").exists()
    styles = list(STYLES)
    if synthetic:
        styles[0] = ("Synthetic observations", *styles[0][1:])
    audit = []
    with plt.rc_context({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 12,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(ceil((len(temperatures) + 1) / 3), 3,
                                 figsize=(12.6, 3.1 * ceil((len(temperatures) + 1) / 3)), squeeze=False)
        fig.suptitle(("SYNTHETIC DEMO: " if synthetic else "") + rf"C$_{{{carbon}}}$H$_{{{2 * carbon + 2}}}$ + N$_2$", fontsize=25)
        for ax, temperature in zip(axes.flat, temperatures):
            # Draw experiment last so a close model fit does not hide the red trace.
            for index in [1, 2, 3, 4, 0]:
                frame, p_col, y_col = traces[index]
                subset = frame.loc[np.isclose(frame["T (K)"], temperature)]
                segments, excluded = trace_segments(subset, p_col, y_col, index != 0)
                audit.extend({"model": styles[index][0], **row} for row in excluded)
                _, color, style, width = styles[index]
                for segment in segments:
                    for composition in ["x_N2", y_col]:
                        x, pressure = smooth_branch(segment[composition].to_numpy(float), segment[p_col].to_numpy(float))
                        ax.plot(x, pressure, color=color, linestyle=style, linewidth=width, zorder=6 if index == 0 else index)
            ax.set_title(rf"$T_r={temperature / tc:.4f}$", fontsize=17, pad=3)
            ax.set_xlabel(r"$x_1,\ y_1$", fontsize=14)
            ax.set_ylabel("P / MPa", fontsize=13)
            ax.minorticks_on()
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
        legend = axes.flat[len(temperatures)]
        legend.axis("off")
        legend.legend(handles=[Line2D([0], [0], label=label, color=color, linestyle=style, linewidth=width)
                               for label, color, style, width in styles], loc="center", fontsize=10, frameon=False)
        for ax in list(axes.flat)[len(temperatures) + 1:]:
            ax.axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = output_dir / f"{system}_vle_comparison.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(audit, columns=["model", "T (K)", "x_N2", "pressure", "reason"]).to_csv(
        output_dir / f"{system}_plot_exclusions.csv", index=False)
    return path
