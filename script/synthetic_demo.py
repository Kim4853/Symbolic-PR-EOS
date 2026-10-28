"""Generate fictional inputs for software checks, never for scientific use."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from script.data import PROJECT_ROOT, SYSTEMS


def generate(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new --out directory.")
    data = output_dir / "data"
    for sub in ["experimental", "calibrated", "baselines/zero", "baselines/generalized"]:
        (data / sub).mkdir(parents=True, exist_ok=False)
    properties = [{"Chemical": "N2", "Tc (K)": 130., "Pc (bar)": 35.,
                   "Acentric Factor (CoolProp)": 0.04}]
    predictions = []
    for system in SYSTEMS:
        carbon = int(system.split("N")[0][1:])
        tc = 400. + 20. * carbon
        properties.append({"Chemical": f"C{carbon}", "Tc (K)": tc,
                           "Pc (bar)": 40. - carbon, "Acentric Factor (CoolProp)": carbon / 25.})
        rows = []
        for tr in [.65, .80, .90]:
            for x in np.linspace(.02, .65, 18):
                pressure = (15. + carbon) * (x + x*x) * (1.2 - tr)
                vapor = .72 + .26 * x
                rows.append({"T (K)": tr * tc, "Tr": tr, "x_N2": x,
                             "P_Exp (MPa)": pressure, "y_N2_Exp": vapor,
                             "P_PR_MPa": pressure + .4 * x, "y_PR": vapor - .02 * x})
        frame = pd.DataFrame(rows)
        frame.to_csv(data / "calibrated" / f"{system}.csv", index=False)
        frame[["T (K)", "x_N2", "P_Exp (MPa)", "y_N2_Exp"]].to_csv(
            data / "experimental" / f"{system}.csv", index=False)
        for kind, p_col, y_col, scale in [("zero", "P_zero_MPa", "y_zero", 1.15),
                                         ("generalized", "P_yang_MPa", "y_yang", 1.06)]:
            baseline = frame[["T (K)", "x_N2"]].copy()
            baseline[p_col] = frame["P_Exp (MPa)"] * scale
            baseline[y_col] = np.clip(frame.y_N2_Exp - (scale - 1.) * frame.x_N2, 0., 1.)
            baseline.to_csv(data / "baselines" / kind / f"{system}.csv", index=False)
        proposed = frame.assign(system=system, P_exp_common=frame["P_Exp (MPa)"],
                                P_pred_nn=frame["P_Exp (MPa)"] * 1.01,
                                y_pred_nn=frame.y_N2_Exp - .001,
                                split_x="synthetic", split_y="synthetic")
        predictions.append(proposed)
    pd.DataFrame(properties).to_csv(data / "properties.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "proposed_predictions.csv", index=False)
    (data / "SYNTHETIC_DEMO.json").write_text(json.dumps({
        "warning": "All observations, properties and predictions are fictional analytic examples.",
        "baseline_predictions": "Not computed by EOS solvers.",
        "proposed_predictions": "Not produced by a trained model.",
        "purpose": "File-format, plotting and training software checks only."
    }, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/synthetic_demo")
    folder = generate(parser.parse_args().out)
    print("FICTIONAL SOFTWARE DEMO ONLY. No experimental measurements are used.")
    print("Set SYMBOLIC_PR_EOS_DATA_DIR to:", folder / "data")
    print("Plot input:", folder / "proposed_predictions.csv")


if __name__ == "__main__":
    main()
