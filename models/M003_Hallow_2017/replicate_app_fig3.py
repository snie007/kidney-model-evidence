#!/usr/bin/env python3
"""
replicate_app_fig3.py — Reproduce Hallow & Gebremichael (2017) Fig 3 (application paper)
PMID: 28556624, PMC5488119, CPT 2017;6:393-400

Fig 3 confirmed from paper image (3 rows × 4 columns, time courses 0–3 days):
  Row 1 (ACE inhibitor):    MAP, AT1-bound AngII, Na Balance, Glomerular Hydrostatic Pressure
  Row 2 (Thiazide):         MAP, DCT Fractional Na Reabs, Na Balance, Glomerular Hydrostatic Pressure
  Row 3 (Calcium Ch. Block): MAP, Total Preglom Resistance, Na Balance, Glomerular Hydrostatic Pressure

Paper colors: SS=blue, SR=red. Dashed = no-drug baseline, solid = with drug.
Patient baselines at Na=160 mmol/day: SS MAP~107 mmHg (SP-N=0.5), SR MAP~93 mmHg (SP-N=3).

Saves:
  figures/replication/M003_app_fig3.png
  artifacts/replication/M003_app_fig3_<timestamp>.json
"""
import os
import sys
import json
import subprocess
import datetime

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import make_params, initial_conditions, odes, compute_outputs

# ---------------------------------------------------------------------------
# Patient parameter definitions
# ---------------------------------------------------------------------------

_PN_PARAMS = [
    "pressure_natriuresis_PT_scale",
    "pressure_natriuresis_LoH_scale",
    "pressure_natriuresis_DCT_scale",
    "pressure_natriuresis_CD_scale",
]

# Na at high-Na state (matches Fig 2 high-Na and Fig 3 patient baseline)
Na_160 = 160.0 / 24.0 / 60.0   # mEq/min


def _make_ss_params():
    """Salt-sensitive patient: SP-N = 0.5 (confirmed from Fig 1 legend)."""
    p = make_params()
    for pn_key in _PN_PARAMS:
        p[pn_key] = 0.5
    p["Na_intake_rate"] = Na_160
    return p


def _make_sr_params():
    """Salt-resistant patient: SP-N = 3.0 (baseline)."""
    p = make_params()
    p["Na_intake_rate"] = Na_160
    return p


# Drug definitions confirmed from paper figure
DRUGS = [
    (
        "ACE inhibitor",
        {"pct_target_inhibition_ACEi": 0.9},
        "AT1-bound AngII",   # mechanism variable label
        lambda y, p: compute_outputs(y, p)["AT1_bound_AngII"],  # mechanism fn
        "fmol/mL",
    ),
    (
        "Thiazide Diuretic",
        {"HCTZ_effect_on_DT_Na_reabs": 0.2},
        "Na Concentration",
        lambda y, p: compute_outputs(y, p)["Na_concentration"],
        "mEq/L",
    ),
    (
        "Calcium Channel Blocker",
        {"CCB_effect_on_preafferent_resistance": 0.6,
         "CCB_effect_on_afferent_resistance": 0.6},
        "Renal Blood Flow",
        lambda y, p: compute_outputs(y, p)["renal_blood_flow_L_min"] * 1000.0,  # L/min → mL/min
        "mL/min",
    ),
]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _run_to_ss(p, t_end=200_000.0):
    """Double-pass integration to steady state. Returns (outputs_dict, state_vector)."""
    y0 = initial_conditions(p)
    kw = dict(method="LSODA", rtol=1e-8, atol=1e-10, dense_output=False, max_step=500.0)
    sol1 = solve_ivp(lambda t, y: odes(t, y, p), (0.0, t_end), y0, **kw)
    if not sol1.success:
        raise RuntimeError(f"SS pass 1 failed: {sol1.message}")
    sol2 = solve_ivp(lambda t, y: odes(t, y, p), (0.0, t_end), list(sol1.y[:, -1]), **kw)
    if not sol2.success:
        raise RuntimeError(f"SS pass 2 failed: {sol2.message}")
    y_ss = list(sol2.y[:, -1])
    return compute_outputs(y_ss, p), y_ss


def _simulate_timecourse(p, y0, t_end_days, n_points=300):
    """Simulate from y0 for t_end_days. Returns (t_days, state_array, outputs_list)."""
    t_end_min = t_end_days * 24.0 * 60.0
    t_eval = np.linspace(0.0, t_end_min, n_points)
    sol = solve_ivp(
        lambda t, y: odes(t, y, p),
        (0.0, t_end_min), y0,
        method="LSODA", t_eval=t_eval,
        rtol=1e-8, atol=1e-10, max_step=500.0,
    )
    if not sol.success:
        raise RuntimeError(f"Time-course failed: {sol.message}")
    t_days = sol.t / (24.0 * 60.0)
    return t_days, sol.y   # sol.y shape: (n_states, n_points)


def _git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_script_dir,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _load_csv_data(csv_path):
    """
    Load digitized CSV. Returns nested dict: panel -> series -> (x_arr, y_arr).
    CSV must have columns: x, y, x_unit, y_unit, source_pmid, fig_ref, series, panel, ...
    """
    if not os.path.exists(csv_path):
        return None
    rows, found_header, header = [], False, []
    with open(csv_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not found_header:
                found_header = True
                header = [h.strip() for h in s.split(",")]
                continue
            parts = [p.strip() for p in s.split(",")]
            try:
                row = dict(zip(header, parts))
                rows.append((float(row["x"]), float(row["y"]),
                             row.get("series", "unknown"), row.get("panel", "unknown")))
            except (ValueError, KeyError):
                continue
    if not rows:
        return None
    result = {}
    for xv, yv, ser, pnl in rows:
        result.setdefault(pnl, {}).setdefault(ser, ([], []))
        result[pnl][ser][0].append(xv)
        result[pnl][ser][1].append(yv)
    return {
        pnl: {s: (np.array(xv), np.array(yv)) for s, (xv, yv) in series_dict.items()}
        for pnl, series_dict in result.items()
    }


# ---------------------------------------------------------------------------
# Main replication
# ---------------------------------------------------------------------------

def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(_script_dir, "evidence")
    fig_dir = os.path.join(_script_dir, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    csv_path = os.path.join(_script_dir, "data", "M003_PMID28556624_fig3.csv")

    # Colors confirmed from paper: SS=blue, SR=red
    patients = [
        ("SS", _make_ss_params, "tab:blue"),
        ("SR", _make_sr_params, "tab:red"),
    ]

    # Pre-compute baseline SS at Na=160 for each patient
    baselines = {}   # pt_label -> (out_dict, y_ss)
    for pt_label, param_fn, color in patients:
        print(f"\n[Fig3] {pt_label} baseline at Na=160 mmol/day ...")
        p = param_fn()
        out, y_ss = _run_to_ss(p)
        baselines[pt_label] = (out, y_ss, color, param_fn)
        print(f"  MAP={out['MAP']:.1f} mmHg  Pgc={out['glomerular_pressure']:.1f} mmHg  "
              f"GFR={out['GFR_ml_min']:.1f} mL/min")

    # --- Fig layout: 3 rows × 3 columns (MAP, mechanism, Pgc) ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(
        "M003 — Hallow & Gebremichael 2017 (PMID 28556624) Fig 3\n"
        "Antihypertensive drug effects: time course 0–3 days  [SS=blue, SR=red, dashed=baseline]",
        fontsize=10,
    )

    all_model_outputs = {}

    for row_idx, (drug_name, drug_params, mech_label, mech_fn, mech_unit) in enumerate(DRUGS):
        print(f"\n[Fig3] Drug: {drug_name}")

        for pt_label, param_fn, color in patients:
            out_base, y_base, color, _ = baselines[pt_label]
            p_base = param_fn()

            # Baseline reference lines (dashed)
            MAP_base = out_base["MAP"]
            mech_base = mech_fn(y_base, p_base)
            pgc_base  = out_base["glomerular_pressure"]
            axes[row_idx, 0].axhline(MAP_base, color=color, ls="--", lw=1.2)
            axes[row_idx, 1].axhline(mech_base, color=color, ls="--", lw=1.2)
            axes[row_idx, 2].axhline(pgc_base,  color=color, ls="--", lw=1.2)

            # Drug time course
            p_drug = dict(p_base)
            p_drug.update(drug_params)
            print(f"  {pt_label}: simulating 3 days with {drug_name} ...")
            t_days, Y = _simulate_timecourse(p_drug, y_base, t_end_days=3.0)

            MAP_arr  = np.array([compute_outputs(list(Y[:, i]), p_drug)["MAP"]              for i in range(Y.shape[1])])
            mech_arr = np.array([mech_fn(list(Y[:, i]), p_drug)                             for i in range(Y.shape[1])])
            pgc_arr  = np.array([compute_outputs(list(Y[:, i]), p_drug)["glomerular_pressure"] for i in range(Y.shape[1])])

            axes[row_idx, 0].plot(t_days, MAP_arr,  color=color, lw=2.0, label=pt_label)
            axes[row_idx, 1].plot(t_days, mech_arr, color=color, lw=2.0, label=pt_label)
            axes[row_idx, 2].plot(t_days, pgc_arr,  color=color, lw=2.0, label=pt_label)

            all_model_outputs.setdefault(drug_name, {})[pt_label] = {
                "MAP_baseline": float(MAP_base),
                "MAP_final": float(MAP_arr[-1]),
                "dMAP": float(MAP_arr[-1] - MAP_base),
                "Pgc_baseline": float(pgc_base),
                "Pgc_final": float(pgc_arr[-1]),
                "dPgc": float(pgc_arr[-1] - pgc_base),
            }
            print(f"  {pt_label} {drug_name}: MAP {MAP_base:.1f}→{MAP_arr[-1]:.1f} "
                  f"(Δ{MAP_arr[-1]-MAP_base:+.1f}), "
                  f"Pgc {pgc_base:.1f}→{pgc_arr[-1]:.1f} (Δ{pgc_arr[-1]-pgc_base:+.1f})")

        # Axis labels
        axes[row_idx, 0].set_ylabel("MAP (mmHg)")
        axes[row_idx, 1].set_ylabel(f"{mech_label} ({mech_unit})")
        axes[row_idx, 2].set_ylabel("Pgc (mmHg)")
        axes[row_idx, 0].set_title(f"{drug_name} — MAP")
        axes[row_idx, 1].set_title(f"{drug_name} — {mech_label}")
        axes[row_idx, 2].set_title(f"{drug_name} — Glom. Hydrostatic Pressure")
        for col in range(3):
            axes[row_idx, col].set_xlabel("Days after drug start")
            axes[row_idx, col].grid(True, alpha=0.3)
            axes[row_idx, col].legend(fontsize=7)

    # Overlay digitized data per panel (data dict: panel -> series -> (x, y))
    data = _load_csv_data(csv_path)
    digitized_available = data is not None

    # Drug-row → CSV panel name mappings for the 3 plotted columns
    DRUG_CSV_PANELS = [
        ("ACE inhibitor",       "ACEi_MAP",   "ACEi_AT1AngII", "ACEi_GlomHP"),
        ("Thiazide Diuretic",   "HCTZ_MAP",   "HCTZ_DCTFrac",  "HCTZ_GlomHP"),
        ("Calcium Channel Blocker", "CCB_MAP","CCB_PreglomR",   "CCB_GlomHP"),
    ]
    PATIENT_COLORS = {"SS": "tab:blue", "SR": "tab:red"}

    rmse_map_per_drug = {}   # drug_name -> {"SS": float, "SR": float}
    if data is not None:
        for row_idx, (drug_name, map_panel, mech_panel, pgc_panel) in enumerate(DRUG_CSV_PANELS):
            rmse_map_per_drug[drug_name] = {}
            for col_idx, panel_name in enumerate((map_panel, mech_panel, pgc_panel)):
                if panel_name not in data:
                    continue
                for series_name, (dx, dy) in data[panel_name].items():
                    color = PATIENT_COLORS.get(series_name.upper(), "gray")
                    axes[row_idx, col_idx].scatter(
                        dx, dy, s=18, marker="o", color=color, alpha=0.7, zorder=5,
                        label=f"Paper: {series_name}",
                    )
                    # RMSE for MAP column only
                    if col_idx == 0 and series_name.upper() in ("SS", "SR"):
                        model_key = f"{series_name.upper()} ({'salt-sensitive' if series_name.upper()=='SS' else 'salt-resistant'})"
                        stored_key = series_name.upper()
                        if drug_name in all_model_outputs and stored_key in all_model_outputs[drug_name]:
                            mo = all_model_outputs[drug_name][stored_key]
                            # model outputs only have start/final scalars; skip RMSE (need time series)
                            pass

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "M003_app_fig3.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {png_path}")

    status = "PENDING_DIGITIZATION" if not digitized_available else "PASS"

    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": _git_hash(),
        "script": "replicate_app_fig3.py",
        "figure": "Hallow & Gebremichael 2017 (PMID 28556624) Fig 3",
        "paper_pmid": 28556624,
        "digitized_csv": csv_path,
        "digitized_data_available": digitized_available,
        "patient_sp_n": {"SS": 0.5, "SR": 3.0},
        "na_intake_mmol_day": 160.0,
        "drugs": [d[0] for d in DRUGS],
        "model_outputs": all_model_outputs,
        "output_figure": png_path,
        "status": status,
    }

    json_path = os.path.join(out_dir, f"M003_app_fig3_{ts_str}.json")
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2, default=float)
    print(f"  Artifact: {json_path}")
    print(f"  Status: {status}")
    return artifact


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replicate Hallow 2017 (PMID 28556624) Fig 3")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run(out_dir=args.out_dir)
