#!/usr/bin/env python3
"""
replicate_fig4b.py — Reproduce Gebremichael 2018 (PMID 29126144) Figure 4B.

Fig 4B: Model fit to cisplatin biomarker responses at 1.0 mg/kg in rat.
Same panels as Fig 4A but at the lower (non-lethal) dose.

Pass criteria (1.0 mg/kg):
  Kim-1 peak fold within 35% of Palmans data peak
  aGST  peak fold within 35% of Palmans data peak

Saves:
  figures/replication/M005_fig4b.png
  artifacts/replication/M005_fig4b_<timestamp>.json
"""
import os, sys, json, datetime, subprocess
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import run_cisplatin, THETA


def _git_hash():
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], cwd=_script_dir,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _load_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        header = True
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if header:
                header = False
                continue
            parts = s.split(",")
            if len(parts) >= 8:
                try:
                    rows.append((float(parts[0]), float(parts[1]), parts[6].strip()))
                except ValueError:
                    continue
    if not rows:
        return None
    xs = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    series = [r[2] for r in rows]
    return xs, ys, series


def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(_script_dir, "evidence")
    fig_dir = os.path.join(_script_dir, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    dig_dir = os.path.join(_script_dir, "data")

    print("[M005 Fig4B] Running model at 1.0 mg/kg cisplatin ...")
    sim = run_cisplatin(dose_mg_kg=1.0, bw_kg=0.25)
    t   = sim["t_days"]

    csv_kim1 = os.path.join(dig_dir, "M005_PMID29126144_fig4b_kim1.csv")
    csv_agst = os.path.join(dig_dir, "M005_PMID29126144_fig4b_agst.csv")
    csv_scr  = os.path.join(dig_dir, "M005_PMID29126144_fig4b_scr.csv")
    d_kim1 = _load_csv(csv_kim1)
    d_agst = _load_csv(csv_agst)
    d_scr  = _load_csv(csv_scr)
    any_data = any(d is not None for d in [d_kim1, d_agst, d_scr])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "M005 — Gebremichael 2018 (PMID 29126144) Fig 4B: Cisplatin 1.0 mg/kg\n"
        "Model vs Palmans data (red diamonds, digitized from Fig 4B)",
        fontsize=10
    )

    # PK panel
    ax = axes[0, 0]
    t_hr = t * 24
    early = t_hr <= 12
    ax.plot(t_hr[early], sim["C_plasma"][early], "b-", lw=1.8, label="C_plasma")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cisplatin (ng/mL)")
    ax.set_title("Cisplatin PK (C_plasma)")
    ax.set_xlim(0, 12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cell fractions
    ax = axes[1, 0]
    ax.plot(t, sim["Fcell"],    "b-",  lw=1.8, label="Functional (F)")
    ax.plot(t, sim["frac_Inj"], "r-",  lw=1.8, label="Injured (I)")
    ax.plot(t, sim["frac_Nec"], "k-",  lw=1.8, label="Dead (N)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cell fraction")
    ax.set_title("Cell Fractions")
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Kim-1
    ax = axes[0, 1]
    ax.plot(t, sim["fold_Kim1"], "b-", lw=1.8, label="Model")
    if d_kim1 is not None:
        xs, ys, ss = d_kim1
        mask = np.array(["Palmans_B_Kim1" in s or "Palmans_Kim1" in s for s in ss])
        ax.scatter(xs[mask], ys[mask], s=40, marker="D", facecolors="none",
                   edgecolors="red", linewidths=1.5, zorder=5, label="Palmans data")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fold Increase")
    ax.set_title("Kim-1 Excretion")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # aGST
    ax = axes[0, 2]
    ax.plot(t, sim["fold_aGST"], "b-", lw=1.8, label="Model")
    if d_agst is not None:
        xs, ys, ss = d_agst
        mask = np.array(["Palmans_B_aGST" in s or "Palmans_aGST" in s for s in ss])
        pos = mask & (ys > 0)
        ax.scatter(xs[pos], ys[pos], s=40, marker="D", facecolors="none",
                   edgecolors="red", linewidths=1.5, zorder=5, label="Palmans data")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fold Increase")
    ax.set_title("aGST Excretion")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # sCr
    ax = axes[1, 2]
    ax.plot(t, sim["fold_sCr"], "b-", lw=1.8, label="Model")
    if d_scr is not None:
        xs, ys, ss = d_scr
        mask_p = np.array(["Palmans" in s for s in ss])
        mask_f = np.array(["Fukushima" in s for s in ss])
        if mask_p.any():
            ax.scatter(xs[mask_p], ys[mask_p], s=40, marker="D", facecolors="none",
                       edgecolors="red", linewidths=1.5, zorder=5, label="Palmans data")
        if mask_f.any():
            ax.scatter(xs[mask_f], ys[mask_f], s=40, marker="^", facecolors="none",
                       edgecolors="green", linewidths=1.5, zorder=5, label="Fukushima data")
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fold Increase")
    ax.set_title("Serum Creatinine (EXCLUDED from PASS)")
    ax.set_xlim(0, 22)
    ax.set_ylim(0.5, 2.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # GFR
    ax = axes[1, 1]
    ax.plot(t, sim["GFR_mlmin"], "b-", lw=1.8, label="GFR (model)")
    ax.axhline(THETA["GFR0_mlmin"], color="gray", lw=0.8, ls="--", label="Baseline")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("GFR (mL/min)")
    ax.set_title("GFR (model only)")
    ax.set_xlim(0, 22)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "M005_fig4b.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {png_path}")

    # RMSE / pass
    peak_kim1_model = float(np.max(sim["fold_Kim1"]))
    peak_agst_model = float(np.max(sim["fold_aGST"]))
    peak_scr_model  = float(np.max(sim["fold_sCr"]))

    peak_kim1_data = None
    peak_agst_data = None
    pct_kim1 = None
    pct_agst = None

    if d_kim1 is not None:
        xs, ys, ss = d_kim1
        mask = np.array(["Kim1" in s for s in ss])
        if mask.any():
            peak_kim1_data = float(ys[mask].max())
            pct_kim1 = abs(peak_kim1_model - peak_kim1_data) / peak_kim1_data * 100

    if d_agst is not None:
        xs, ys, ss = d_agst
        mask = np.array(["aGST" in s for s in ss]) & (ys > 1.0)
        if mask.any():
            peak_agst_data = float(ys[mask].max())
            pct_agst = abs(peak_agst_model - peak_agst_data) / peak_agst_data * 100

    PASS_THRESH_PCT = 35.0
    pass_kim1 = (pct_kim1 is not None and pct_kim1 < PASS_THRESH_PCT)
    pass_agst = (pct_agst is not None and pct_agst < PASS_THRESH_PCT)

    if not any_data or (d_kim1 is None and d_agst is None):
        status = "PENDING_DIGITIZATION"
    else:
        status = "PASS" if (pass_kim1 and pass_agst) else "FAIL"

    print(f"  Kim-1: model {peak_kim1_model:.1f}x, data {peak_kim1_data}x, "
          f"diff {pct_kim1}% -> {'PASS' if pass_kim1 else 'FAIL'}")
    print(f"  aGST:  model {peak_agst_model:.1f}x, data {peak_agst_data}x, "
          f"diff {pct_agst}% -> {'PASS' if pass_agst else 'FAIL'}")
    print(f"  sCr:   model {peak_scr_model:.2f}x (EXCLUDED)")
    print(f"  Status: {status}")

    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": _git_hash(),
        "script": "replicate_fig4b.py",
        "figure": "Gebremichael 2018 (PMID 29126144) Fig 4B",
        "paper_pmid": 29126144,
        "dose_mg_kg": 1.0,
        "digitized_csvs": {"kim1": csv_kim1, "agst": csv_agst, "scr": csv_scr},
        "model_peaks": {
            "Kim1_fold":     peak_kim1_model,
            "aGST_fold":     peak_agst_model,
            "sCr_fold":      peak_scr_model,
            "GFR_min_mlmin": float(np.min(sim["GFR_mlmin"])),
            "Inj_max":       float(np.max(sim["frac_Inj"])),
            "Nec_max":       float(np.max(sim["frac_Nec"])),
        },
        "data_peaks": {
            "Kim1_fold_Palmans": peak_kim1_data,
            "aGST_fold_Palmans": peak_agst_data,
        },
        "pct_error":          {"Kim1": pct_kim1, "aGST": pct_agst, "sCr": None},
        "pass_threshold_pct": PASS_THRESH_PCT,
        "pass_by_criterion":  {"Kim1": pass_kim1, "aGST": pass_agst, "sCr": "EXCLUDED_KNOWN_GAP"},
        "status":             status,
        "output_figure":      png_path,
    }

    json_path = os.path.join(out_dir, f"M005_fig4b_{ts_str}.json")
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
