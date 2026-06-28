"""
Replicate Mahato et al. 2018 Figure 5 — DKD Disease Progression.

Figure 5 panels (model predictions vs experimental data from Fig 2):
  A — Glomerular pressure over time (db/db, db/db UNX, non-db control)
  B — Kf (adaptive filtration surface) over time
  C — Number of functional nephrons over time
  D — MAP over time
  E — GFR over time (compared against Fig 2B experimental data)
  F — UAER over time (compared against Fig 2C experimental data)

Produces:
  evidence/M018_fig5_YYYYMMDD_HHMMSS.json
  evidence/M018_fig5_YYYYMMDD_HHMMSS.png

PMID: 30564457
"""

import os
import sys
import json
import time
import datetime
import subprocess
import numpy as np
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from model import make_params, run_to_ss, simulate_dkd, compute_outputs

ARTIFACTS_DIR = _script_dir / "evidence"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = _script_dir / "data"
PMID = "30564457"


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_script_dir), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_csv(path):
    import csv
    if not Path(path).exists():
        return {}
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["series"]
            x, y = float(row["x"]), float(row["y"])
            if s not in data:
                data[s] = []
            data[s].append((x, y))
    return data


def median_by_week(points, week_bins=None):
    if not points:
        return np.array([]), np.array([])
    if week_bins is None:
        week_bins = np.arange(4, 24, 2)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    out_x, out_y = [], []
    for wb in week_bins:
        near = np.abs(xs - wb) < 1.5
        if near.sum() >= 1:
            out_x.append(wb)
            out_y.append(np.median(ys[near]))
    return np.array(out_x), np.array(out_y)


def rmse_vs_data(model_t, model_y, data_t, data_y):
    if len(data_t) == 0:
        return np.nan
    model_interp = np.interp(data_t, model_t, model_y)
    data_mean = np.mean(data_y)
    if abs(data_mean) < 1e-12:
        return np.nan
    return float(np.sqrt(np.mean((model_interp - data_y)**2)) / abs(data_mean) * 100.0)


def run_replication():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()

    print("=" * 60)
    print("M018 Mahato 2018 — Fig 5 Replication")
    print(f"Timestamp: {timestamp}   git: {git_hash}")
    print("=" * 60)

    csv_GFR  = DATA_DIR / f"M018_PMID{PMID}_fig2b_GFR.csv"
    csv_UAER = DATA_DIR / f"M018_PMID{PMID}_fig2c_UAER.csv"

    gfr_data  = load_csv(csv_GFR)
    uaer_data = load_csv(csv_UAER)

    gfr_dbdb_pts  = gfr_data.get("dbdb", [])
    uaer_dbdb_pts = uaer_data.get("dbdb", [])
    gfr_ctrl_pts  = gfr_data.get("WT_or_dbm", [])

    gfr_dbdb_t,  gfr_dbdb_y  = median_by_week(gfr_dbdb_pts)
    uaer_dbdb_t, uaer_dbdb_y = median_by_week(uaer_dbdb_pts)
    gfr_ctrl_t,  gfr_ctrl_y  = median_by_week(gfr_ctrl_pts)

    print(f"  Loaded {len(gfr_dbdb_pts)} GFR (db/db) and {len(uaer_dbdb_pts)} UAER (db/db) data points")

    print("\n  Running lean control steady-state ...")
    t_start = time.time()
    p_ctrl = make_params("normal")
    _, y_ss = run_to_ss(p_ctrl)
    out_ctrl = compute_outputs(y_ss, p_ctrl)
    t_ctrl = time.time() - t_start
    print(f"    GFR = {out_ctrl['GFR_mL_min']:.3f} mL/min  MAP = {out_ctrl['MAP_mmHg']:.1f} mmHg")

    print("\n  Running db/db DKD simulation (25 weeks) ...")
    t_start = time.time()
    p_dbdb = make_params("dbdb")
    _, outs_dbdb = simulate_dkd("dbdb", t_end_weeks=25.0, p=p_dbdb, y0_ss=y_ss)
    t_dbdb = time.time() - t_start
    print(f"    Done in {t_dbdb:.1f}s  GFR {outs_dbdb['GFR_mL_min'][0]:.3f}→{outs_dbdb['GFR_mL_min'][-1]:.3f} mL/min")

    print("\n  Running db/db UNX simulation (25 weeks) ...")
    t_start = time.time()
    p_unx = make_params("dbdb_unx")
    _, outs_unx = simulate_dkd("dbdb_unx", t_end_weeks=25.0, p=p_unx, y0_ss=y_ss)
    t_unx = time.time() - t_start
    print(f"    Done in {t_unx:.1f}s  GFR {outs_unx['GFR_mL_min'][0]:.3f}→{outs_unx['GFR_mL_min'][-1]:.3f} mL/min")

    t_weeks = outs_dbdb["t_weeks"]
    rmse_GFR_dbdb  = rmse_vs_data(t_weeks, outs_dbdb["GFR_mL_min"], gfr_dbdb_t, gfr_dbdb_y)
    rmse_UAER_dbdb = rmse_vs_data(t_weeks, outs_dbdb["UAER_ug_day"], uaer_dbdb_t, uaer_dbdb_y)
    rmse_GFR_ctrl  = rmse_vs_data(t_weeks, np.full_like(t_weeks, out_ctrl["GFR_mL_min"]),
                                  gfr_ctrl_t, gfr_ctrl_y)

    gfr_peak   = float(np.max(outs_dbdb["GFR_mL_min"]))
    gfr_ctrl_v = float(out_ctrl["GFR_mL_min"])
    uaer_final = float(outs_dbdb["UAER_ug_day"][-1])
    uaer_init  = float(outs_dbdb["UAER_ug_day"][0])
    hyperfiltration_present = gfr_peak > gfr_ctrl_v * 1.05
    uaer_rising = uaer_final > uaer_init * 1.1

    if np.isnan(rmse_GFR_dbdb):
        status = "INFORMATIVE_PASS" if (hyperfiltration_present and uaer_rising) else "INFORMATIVE_FAIL"
        reason = "No digitized data; trends checked"
    elif rmse_GFR_dbdb <= 30.0 and uaer_rising:
        status = "PASS"
        reason = f"GFR RMSE={rmse_GFR_dbdb:.1f}% ≤ 30%; UAER rising"
    elif rmse_GFR_dbdb <= 50.0 and uaer_rising:
        status = "INFORMATIVE_PASS"
        reason = f"GFR RMSE={rmse_GFR_dbdb:.1f}% (30-50%); UAER rising"
    else:
        status = "FAIL"
        reason = f"GFR RMSE={rmse_GFR_dbdb:.1f}% > 50% or UAER not rising"

    print(f"\n  *** {status}: {reason} ***")

    artifact = {
        "timestamp":    timestamp,
        "git_commit":   git_hash,
        "model":        "M018_Mahato_2018",
        "figure":       "fig5",
        "pmid":         PMID,
        "status":       status,
        "reason":       reason,
        "input_csvs":   {"GFR": str(csv_GFR), "UAER": str(csv_UAER)},
        "baseline_outputs": {k: float(v) for k, v in out_ctrl.items()
                             if isinstance(v, (int, float, np.floating, np.integer))},
        "rmse_pct": {
            "GFR_dbdb":  float(rmse_GFR_dbdb)  if not np.isnan(rmse_GFR_dbdb)  else None,
            "UAER_dbdb": float(rmse_UAER_dbdb) if not np.isnan(rmse_UAER_dbdb) else None,
            "GFR_ctrl":  float(rmse_GFR_ctrl)  if not np.isnan(rmse_GFR_ctrl)  else None,
        },
        "qualitative_checks": {
            "hyperfiltration_present": hyperfiltration_present,
            "uaer_rising":             uaer_rising,
        },
    }

    json_path = ARTIFACTS_DIR / f"M018_fig5_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n  Artifact: {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f"M018 Mahato 2018 — Fig 5 DKD Progression [{status}]\n"
            f"PMID {PMID}  git:{git_hash}  {timestamp}", fontsize=11)

        t_w   = outs_dbdb["t_weeks"]
        t_w_u = outs_unx["t_weeks"]

        axes[0,0].axhline(out_ctrl["Pgc_mmHg"], color="black", lw=2, label="non-db")
        axes[0,0].plot(t_w, outs_dbdb["Pgc_mmHg"], color="steelblue", lw=2, label="db/db")
        axes[0,0].plot(t_w_u, outs_unx["Pgc_mmHg"], color="firebrick", lw=2, label="db/db UNX")
        axes[0,0].set(xlabel="Time (weeks)", ylabel="Pgc (mmHg)", title="A: Glomerular Pressure")
        axes[0,0].legend(fontsize=8)

        axes[0,1].axhline(out_ctrl["Kf_nL_min_mmHg"], color="black", lw=2, label="non-db")
        axes[0,1].plot(t_w, outs_dbdb["Kf_nL_min_mmHg"], color="steelblue", lw=2, label="db/db")
        axes[0,1].plot(t_w_u, outs_unx["Kf_nL_min_mmHg"], color="firebrick", lw=2, label="db/db UNX")
        axes[0,1].set(xlabel="Time (weeks)", ylabel="Kf (nL/min/mmHg)", title="B: Kf")
        axes[0,1].legend(fontsize=8)

        axes[0,2].axhline(out_ctrl["N_nephrons"], color="black", lw=2, ls="--", label="non-db")
        axes[0,2].plot(t_w, outs_dbdb["N_nephrons"], color="steelblue", lw=2, label="db/db")
        axes[0,2].plot(t_w_u, outs_unx["N_nephrons"], color="firebrick", lw=2, label="db/db UNX")
        axes[0,2].set(xlabel="Time (weeks)", ylabel="Functional nephrons", title="C: Nephron Number")
        axes[0,2].legend(fontsize=8)

        axes[1,0].axhline(out_ctrl["MAP_mmHg"], color="black", lw=2, label="non-db")
        axes[1,0].plot(t_w, outs_dbdb["MAP_mmHg"], color="steelblue", lw=2, label="db/db")
        axes[1,0].plot(t_w_u, outs_unx["MAP_mmHg"], color="firebrick", lw=2, label="db/db UNX")
        axes[1,0].set(xlabel="Time (weeks)", ylabel="MAP (mmHg)", title="D: Mean Arterial Pressure")
        axes[1,0].legend(fontsize=8)

        axes[1,1].axhline(out_ctrl["GFR_mL_min"], color="black", lw=2, label="non-db (model)")
        axes[1,1].plot(t_w, outs_dbdb["GFR_mL_min"], color="steelblue", lw=2, label="db/db (model)")
        axes[1,1].plot(t_w_u, outs_unx["GFR_mL_min"], color="firebrick", lw=2, label="db/db UNX (model)")
        if len(gfr_dbdb_t) > 0:
            axes[1,1].scatter(gfr_dbdb_t, gfr_dbdb_y, color="steelblue", marker="^", zorder=5, s=40, label="db/db data")
        if len(gfr_ctrl_t) > 0:
            axes[1,1].scatter(gfr_ctrl_t, gfr_ctrl_y, color="black", marker="o", zorder=5, s=40, label="ctrl data")
        axes[1,1].set(xlabel="Time (weeks)", ylabel="GFR (mL/min)",
                      title=f"E: GFR  [RMSE={rmse_GFR_dbdb:.1f}%]")
        axes[1,1].legend(fontsize=7)

        axes[1,2].plot(t_w, outs_dbdb["UAER_ug_day"], color="steelblue", lw=2, label="db/db (model)")
        axes[1,2].plot(t_w_u, outs_unx["UAER_ug_day"], color="firebrick", lw=2, label="db/db UNX (model)")
        if len(uaer_dbdb_t) > 0:
            axes[1,2].scatter(uaer_dbdb_t, uaer_dbdb_y, color="steelblue", marker="^", zorder=5, s=40, label="db/db data")
        axes[1,2].set(xlabel="Time (weeks)", ylabel="UAER (µg/day)",
                      title=f"F: UAER  [RMSE={rmse_UAER_dbdb:.1f}%]")
        axes[1,2].legend(fontsize=7)

        plt.tight_layout()
        plot_path = ARTIFACTS_DIR / f"M018_fig5_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {plot_path}")

    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    print(f"\n  FINAL STATUS: {status}")
    return artifact


if __name__ == "__main__":
    result = run_replication()
    sys.exit(0 if result["status"] in ("PASS", "INFORMATIVE_PASS") else 1)
