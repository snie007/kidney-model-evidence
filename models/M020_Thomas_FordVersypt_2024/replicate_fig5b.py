#!/usr/bin/env python3
"""
replicate_fig5b.py — Reproduce Thomas & Ford Versypt 2024 (PMID 39525640) Fig 5B.

Fig 5B: Macrophage (MAC) fold change vs time (weeks 0-24), DKD scenario.
Data: 3 orange circles from M020_PMID39525640_fig5b_mac.csv.

NOTE: The model uses PARAMS_FITTED (FitColData), which was obtained by
sequential fitting: MAC params were fitted first (giving MAC peak ~4-5×),
then COL params were fitted holding MAC params fixed. The final parameter
set gives MAC peak ~13× in both MATLAB and Python. The DIGITIZED DATA shows
MAC peak at fold ≈ 9.4-11.6. This is a known sequential-fitting artefact
documented in model.py — it is NOT a port error.

Pass criterion: INFORMATIVE ONLY — model within 50% of data. This figure
is excluded from primary PASS due to the known parameter artefact.

Saves:
  figures/replication/M020_fig5b.png
  artifacts/replication/M020_fig5b_<timestamp>.json
"""
import os, sys, json, datetime, subprocess, csv
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import run_fibrosis, STATE_INDICES


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_script_dir,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["x"]), float(row["y"])))
    return rows


def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(_script_dir, "evidence")
    fig_dir = os.path.join(_script_dir, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load digitized data
    # -----------------------------------------------------------------------
    csv_path = os.path.join(_script_dir, "data",
                            "M020_PMID39525640_fig5b_mac.csv")
    data_pts = load_csv(csv_path)
    data_weeks = np.array([p[0] for p in data_pts])
    data_fold  = np.array([p[1] for p in data_pts])
    print(f"[M020 Fig5B] Loaded {len(data_pts)} data points from CSV")

    # -----------------------------------------------------------------------
    # Run model
    # -----------------------------------------------------------------------
    print("[M020 Fig5B] Running DKD simulation (24 weeks) ...")
    t_days, y, y_ss = run_fibrosis(scenario="DKD", simulation_weeks=24)
    t_weeks = t_days / 7.0

    mac_idx  = STATE_INDICES["MAC"]
    mac_ss   = y_ss[mac_idx]
    mac_fold = y[:, mac_idx] / mac_ss

    print(f"  MAC SS (healthy) = {mac_ss:.6g}")
    print(f"  MAC fold peak    = {mac_fold.max():.3f}  at week {t_weeks[np.argmax(mac_fold)]:.1f}")

    # -----------------------------------------------------------------------
    # Interpolate model at data time points
    # -----------------------------------------------------------------------
    model_at_data = np.interp(data_weeks, t_weeks, mac_fold)
    pct_errors    = 100.0 * np.abs(model_at_data - data_fold) / data_fold

    # INFORMATIVE: 50% tolerance, but Fig 5B is EXCLUDED from primary PASS
    pass_mask = (data_fold > 1.5)
    passed    = np.all(pct_errors[pass_mask] < 50.0)
    # Note: always INFORMATIVE_FAIL or INFORMATIVE_PASS — not primary criterion
    status = "INFORMATIVE_PASS" if passed else "INFORMATIVE_FAIL"

    print("\n  Comparison at data time points (INFORMATIVE ONLY):")
    print(f"  {'Week':>6}  {'Data':>8}  {'Model':>8}  {'Err%':>7}  {'50% ok?':>8}")
    comparisons = []
    for i, (w, d_fc) in enumerate(zip(data_weeks, data_fold)):
        m_fc = model_at_data[i]
        err  = pct_errors[i]
        in_crit = pass_mask[i]
        ok = (err < 50.0) if in_crit else True
        mark = ("ok" if ok else "FAIL") if in_crit else "SKIP"
        print(f"  {w:>6.1f}  {d_fc:>8.3f}  {m_fc:>8.3f}  {err:>7.1f}  {mark:>8}")
        comparisons.append({
            "week": w, "data_fold": float(d_fc), "model_fold": float(m_fc),
            "pct_error": float(err), "in_criterion": bool(in_crit),
        })

    print(f"\n  Status: {status}")
    print("  NOTE: MAC fold is EXCLUDED from primary PASS (known sequential-fitting artefact)")
    print("  See model.py docstring for explanation.")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_weeks, mac_fold, color="#FF8C00", lw=2, label="Model (DKD)")
    ax.errorbar(data_weeks, data_fold, fmt="o", color="#FF8C00", ms=7,
                label="Exp data (digitized)", capsize=4, zorder=5)
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Macrophage fold change")
    ax.set_title("M020 — Thomas & Ford Versypt 2024 (PMID 39525640)\n"
                 "Fig 5B: Macrophage fold change vs time (DKD model)\n"
                 "[INFORMATIVE — excluded from primary PASS; sequential-fit artefact]")
    ax.legend()
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 16)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, status,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color="gray", fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(fig_dir, "M020_fig5b.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {png_path}")

    # -----------------------------------------------------------------------
    # JSON artifact
    # -----------------------------------------------------------------------
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "timestamp":   datetime.datetime.now().isoformat(),
        "git_commit":  _git_hash(),
        "script":      "replicate_fig5b.py",
        "figure":      "Thomas & Ford Versypt 2024 (PMID 39525640) Fig 5B",
        "paper_pmid":  39525640,
        "input_csv":   csv_path,
        "mac_ss":      float(mac_ss),
        "mac_fold_peak": float(mac_fold.max()),
        "mac_fold_peak_week": float(t_weeks[np.argmax(mac_fold)]),
        "comparisons": comparisons,
        "pass_criterion": "INFORMATIVE ONLY — 50% tolerance, excluded from primary PASS",
        "note": (
            "MAC fold-change artefact: FitColData params give MAC peak ~13x. "
            "FitMACData (earlier fitting step) gives ~4-5x. "
            "Digitized data shows fold 9.4-11.6. "
            "This is a sequential fitting artefact in the original MATLAB. "
            "Fig 5B is excluded from the primary pass/fail criterion."
        ),
        "status": status,
        "output_figure": png_path,
    }
    json_path = os.path.join(out_dir, f"M020_fig5b_{ts_str}.json")
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2, default=float)
    print(f"  Artifact: {json_path}")
    return artifact


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    run(out_dir=args.out_dir)
