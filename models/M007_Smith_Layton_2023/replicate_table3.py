#!/usr/bin/env python3
"""
replicate_table3.py — Reproduce Smith & Layton 2023 (PMID 36952058) Table III.

Table III: Steady-state AngII concentrations in all intrarenal compartments.
Reference source: bioRxiv preprint (doi:10.1101/2021.12.14.472639) Table III.
Note: published paper (PMID 36952058) is paywalled; preprint values match
the MATLAB model code to <0.2%. Published paper reports AngII_circ=14 fmol/mL
with a differently fitted parameter set.

Pass criterion: model SS AngII values within 1% of preprint Table III.

Saves:
  figures/replication/M007_table3.png
  artifacts/replication/M007_table3_<timestamp>.json
"""
import os, sys, json, datetime, subprocess, csv
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import run_baseline, get_params


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_script_dir,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def load_csv(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["x"]] = float(row["y"])
    return rows


# Variables included in the PASS criterion (AngII only, as the table's
# physiologically meaningful validation targets)
PASS_VARS = [
    "AngII_circ",
    "AngII_Isf_Gl",
    "AngII_Isf_Pt",
    "AngII_Fl_Tb",
    "AngII_Pv",
    "AngII_T",
    "AngII_cell_Gl",
    "AngII_cell_Pt",
    "AngII_cell_Tb",
]

# All Table III variables to report (broader than PASS set)
ALL_VARS = [
    "AGT_circ", "AngI_circ", "AngII_circ",
    "AT1R_AngII_memb_circ", "Ang17_circ", "PRC",
    "AngII_Isf_Gl", "AT1R_AngII_memb_Gl", "AT1R_AngII_cell_Gl", "AngII_cell_Gl",
    "AngII_Isf_Pt", "AT1R_AngII_memb_Pt", "AT1R_AngII_cell_Pt", "AngII_cell_Pt",
    "AngII_Fl_Tb", "AT1R_AngII_memb_Tb", "AT1R_AngII_cell_Tb", "AngII_cell_Tb",
    "AngII_Pv", "AT1R_AngII_memb_Pv",
    "AngI_T", "AngII_T",
]

TOLERANCE = 0.01  # 1%


def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(_script_dir, "evidence")
    fig_dir = os.path.join(_script_dir, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load digitized table data
    # -----------------------------------------------------------------------
    csv_path = os.path.join(_script_dir, "data",
                            "M007_PMID36952058_table3_ss.csv")
    ref_data = load_csv(csv_path)
    print(f"[M007 Table3] Loaded {len(ref_data)} reference values from CSV")

    # -----------------------------------------------------------------------
    # Run model
    # -----------------------------------------------------------------------
    print("[M007 Table3] Running 7-day baseline (no Ang II infusion) ...")
    p = get_params()
    result, sol = run_baseline(p, days=7, rtol=1e-8, atol=1e-10)
    print(f"  Solver: {sol.message}, {len(sol.t)} steps")

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    print(f"\n  {'Variable':<30} {'Ref':>12} {'Model':>12} {'Err%':>8} {'Pass?':>7}")
    print("  " + "-" * 75)
    comparisons = []
    pass_mask_results = []

    for var in ALL_VARS:
        if var not in ref_data:
            continue
        ref_val   = ref_data[var]
        model_val = result.get(var, float("nan"))
        if np.isnan(model_val):
            print(f"  {var:<30} {'?':>12}  MISSING IN MODEL")
            continue
        pct_err = 100.0 * abs(model_val - ref_val) / abs(ref_val) if ref_val != 0 else 0.0
        in_pass = var in PASS_VARS
        passed  = pct_err <= (TOLERANCE * 100) if in_pass else None
        mark    = ("PASS" if passed else "FAIL") if in_pass else "INFO"
        print(f"  {var:<30} {ref_val:>12.4g} {model_val:>12.4g} {pct_err:>7.3f}%  {mark}")
        comparisons.append({
            "variable": var, "ref": float(ref_val), "model": float(model_val),
            "pct_error": float(pct_err), "in_pass_criterion": in_pass,
            "pass": bool(passed) if in_pass else None,
        })
        if in_pass:
            pass_mask_results.append(passed)

    print("  " + "-" * 75)
    overall_pass = all(pass_mask_results)
    status = "PASS" if overall_pass else "FAIL"
    n_pass = sum(pass_mask_results)
    n_fail = len(pass_mask_results) - n_pass
    print(f"\n  AngII criterion: {n_pass}/{len(pass_mask_results)} within {TOLERANCE*100:.0f}%")
    print(f"  Overall status: {status}")

    # -----------------------------------------------------------------------
    # Figure: model vs preprint Table III (AngII compartments only)
    # -----------------------------------------------------------------------
    ang_vars_plot = [v for v in PASS_VARS if v in ref_data and v in result]
    ref_vals    = np.array([ref_data[v] for v in ang_vars_plot])
    model_vals  = np.array([result[v] for v in ang_vars_plot])
    short_names = [v.replace("AngII_", "").replace("_", "\n") for v in ang_vars_plot]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ang_vars_plot))
    w = 0.35
    ax.bar(x - w/2, ref_vals, w, label="Preprint Table III", color="#4878D0", alpha=0.8)
    ax.bar(x + w/2, model_vals, w, label="Python model (7-day SS)", color="#EE854A", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Concentration (fmol/mL or fmol/g)")
    ax.set_title("M007 — Smith & Layton 2023 (PMID 36952058)\n"
                 "Table III: Steady-state AngII compartment values\n"
                 "(reference = bioRxiv preprint Table III, matching MATLAB code)")
    ax.set_yscale("log")
    ax.legend()
    ax.text(0.02, 0.98, status,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, color="green" if status == "PASS" else "red",
            fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(fig_dir, "M007_table3.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {png_path}")

    # -----------------------------------------------------------------------
    # JSON artifact
    # -----------------------------------------------------------------------
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "git_commit":   _git_hash(),
        "script":       "replicate_table3.py",
        "figure":       "Smith & Layton 2023 (PMID 36952058) Table III",
        "paper_pmid":   36952058,
        "ref_source":   "bioRxiv preprint doi:10.1101/2021.12.14.472639 Table III",
        "input_csv":    csv_path,
        "comparisons":  comparisons,
        "n_pass":       n_pass,
        "n_fail":       n_fail,
        "pass_criterion": f"AngII in all compartments within {TOLERANCE*100:.0f}% of preprint Table III",
        "note": (
            "Published paper (PMID 36952058) reports AngII_circ=14 fmol/mL with a "
            "re-fitted parameter set. The MATLAB code (and Python port) use preprint "
            "parameters giving AngII_circ=43.4 fmol/mL. This replication validates "
            "the Python port against the preprint Table III (which matches MATLAB output). "
            "The preprint PDF is at resources/papers/S010_Smith_Layton_2023_intrarenal_RAS_preprint.pdf."
        ),
        "status":       status,
        "output_figure": png_path,
    }
    json_path = os.path.join(out_dir, f"M007_table3_{ts_str}.json")
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
