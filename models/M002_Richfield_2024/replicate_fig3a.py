#!/usr/bin/env python3
"""
replicate_fig3a.py — Reproduce Richfield 2024 (PMID 38966231) Figure 3A (partial).

Figure 3A: Q_AA (nl/min) vs Perfusion Pressure (mmHg) from Takenaka 1994.
Data series: Control (autoregulation intact), Furosemide (TGF removed),
Diltiazem (myogenic removed).

Python scope: The Python port implements the GLOMERULAR SURROGATE only, not the
full autoregulation loop (myogenic + TGF feedback). This script validates the
baseline (Pa=100, control condition) where D_AA=7.0 um (Takenaka 1994 measured value).

PASS criterion: Q_blood at Pa=100 (control) within 3% of MATLAB reference.

Reference data source: Q values computed from MATLAB R autoregulation model output
(Myo_TGF_model_curves_20231001.mat, resources/code/autoreg_glommod/) at equilibrium
D_AA values matching Takenaka 1994 measurements. Saved to:
  resources/digitized/M002_PMID38966231_fig3a_control.csv
  resources/digitized/M002_PMID38966231_fig3a_furosemide.csv

Saves:
  figures/replication/M002_fig3a.png
  artifacts/replication/M002_fig3a_<timestamp>.json
"""
import os, sys, json, csv, datetime, subprocess
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import run_baseline_surrogate

HT = 0.40  # haematocrit, Richfield 2024 model default


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_script_dir,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"x": float(row["x"]), "y": float(row["y"])})
    return sorted(rows, key=lambda r: r["x"])


TOLERANCE_PCT = 3.0  # 3% — looser than M007/M020 because only Pa=100 is validated


def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(_script_dir, "evidence")
    fig_dir = os.path.join(_script_dir, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load reference data
    ctrl_path = os.path.join(_script_dir, "data",
                             "M002_PMID38966231_fig3a_control.csv")
    furo_path = os.path.join(_script_dir, "data",
                             "M002_PMID38966231_fig3a_furosemide.csv")
    ctrl_data = load_csv(ctrl_path)
    furo_data = load_csv(furo_path)
    print(f"[M002 Fig3A] Control reference: {len(ctrl_data)} pts, "
          f"Furosemide reference: {len(furo_data)} pts")

    # Run Python model across Pa=100-150 (surrogate, fixed D_AA = 7 um)
    Pa_range = np.arange(100, 155, 5, dtype=float)
    Q_blood_model = []
    for Pa in Pa_range:
        try:
            r = run_baseline_surrogate(Pa=float(Pa))
            Q_blood_model.append(r["Q"] / (1.0 - HT))
        except (ValueError, Exception):
            Q_blood_model.append(np.nan)
    Q_blood_model = np.array(Q_blood_model)

    # Primary comparison: Pa=100 only (model uses D_AA = 7.0 um = Takenaka control)
    ctrl_100 = next(r["y"] for r in ctrl_data if r["x"] == 100.0)
    model_100 = Q_blood_model[0]
    pct_err_100 = abs(model_100 - ctrl_100) / ctrl_100 * 100.0
    passed = pct_err_100 <= TOLERANCE_PCT
    status = "PASS" if passed else "FAIL"

    print(f"\n  Pa=100 control: ref={ctrl_100:.2f} nl/min, model={model_100:.2f} nl/min, "
          f"err={pct_err_100:.2f}%  [{status}]")
    print(f"  Note: Pa=125, 150 autoregulation curve not validated — requires full R model.")

    # Figure
    Pa_ctrl = np.array([r["x"] for r in ctrl_data])
    Q_ctrl  = np.array([r["y"] for r in ctrl_data])
    Pa_furo = np.array([r["x"] for r in furo_data])
    Q_furo  = np.array([r["y"] for r in furo_data])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Pa_ctrl, Q_ctrl, "o", color="black", ms=7, label="Takenaka 1994 — Control (MATLAB ref)")
    ax.plot(Pa_furo, Q_furo, "s", color="dimgray", ms=7, mfc="none",
            label="Takenaka 1994 — Furosemide (MATLAB ref)")
    ax.plot(Pa_range, Q_blood_model, "-", color="red", linewidth=2,
            label=f"Python model — fixed D=7 µm (Q_blood = Q_plasma/(1-Ht))")
    ax.axvline(100, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Perfusion Pressure (mmHg)")
    ax.set_ylabel("Q_AA (nl/min)")
    ax.set_xlim(95, 160)
    ax.set_ylim(100, 600)
    ax.set_title(
        "M002 — Richfield 2024 (PMID 38966231) Figure 3A (partial)\n"
        "Q_AA vs Pa — Python model vs Takenaka 1994 (MATLAB-computed)"
    )
    ax.legend(fontsize=8)
    ax.text(0.02, 0.98, f"Pa=100: {status} ({pct_err_100:.1f}% error)\n"
            "Pa=125/150: not validated (R code needed)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="green" if passed else "red",
            fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(fig_dir, "M002_fig3a.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {png_path}")

    # JSON artifact (Rule 5)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparisons = [
        {
            "Pa_mmHg": 100.0, "condition": "control",
            "ref_Q_blood_nl_per_min": float(ctrl_100),
            "model_Q_blood_nl_per_min": float(model_100),
            "pct_error": float(pct_err_100),
            "pass": bool(passed),
            "note": "Python model uses fixed D_AA=7.0 um (Takenaka control at Pa=100).",
        },
        {
            "Pa_mmHg": 125.0, "condition": "control",
            "ref_Q_blood_nl_per_min": float(next(r["y"] for r in ctrl_data if r["x"] == 125.0)),
            "model_Q_blood_nl_per_min": None,
            "pct_error": None,
            "pass": None,
            "note": "Requires R autoregulation loop (D_AA=6.19 um); not in Python scope.",
        },
        {
            "Pa_mmHg": 150.0, "condition": "control",
            "ref_Q_blood_nl_per_min": float(next(r["y"] for r in ctrl_data if r["x"] == 150.0)),
            "model_Q_blood_nl_per_min": None,
            "pct_error": None,
            "pass": None,
            "note": "Requires R autoregulation loop (D_AA=5.73 um); not in Python scope.",
        },
    ]
    artifact = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "git_commit":     _git_hash(),
        "script":         "replicate_fig3a.py",
        "figure":         "Richfield 2024 (PMID 38966231) Figure 3A — Q_AA vs Pa",
        "paper_pmid":     38966231,
        "ref_source":     "MATLAB Myo_TGF_model_curves_20231001.mat (Q_pred_cont); "
                          "Q_blood = Q_plasma/(1-0.4)",
        "input_csvs":     [ctrl_path, furo_path],
        "comparisons":    comparisons,
        "pass_criterion": f"Pa=100 control Q_blood within {TOLERANCE_PCT:.0f}% of MATLAB reference",
        "n_validated":    1,
        "n_not_in_scope": 5,
        "status":         status,
        "output_figure":  png_path,
        "scope_note": (
            "The Python port implements the glomerular surrogate at fixed D_AA=7.0 um. "
            "Validation of the full autoregulation pressure-flow curve (Pa=100-150 with "
            "varying D_AA from Takenaka 1994) requires the R autoregulation model "
            "(resources/code/autoreg_glommod/). "
            "Only the Pa=100 baseline (D_AA=7.0 um, control condition) is validated here."
        ),
    }
    json_path = os.path.join(out_dir, f"M002_fig3a_{ts_str}.json")
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2, default=float)
    print(f"  Artifact: {json_path}")

    if status != "PASS":
        sys.exit(1)
    return artifact


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    run(out_dir=args.out_dir)
