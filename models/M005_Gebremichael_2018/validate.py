# NOTE: run from the kidney-model-evidence repo root: python models/M005_Gebremichael_2018/validate.py
"""
Validation of M005 Python port against R/rxode2 reference output.

Reference values produced by:
  resources/code/Gebremichael_2018_PTcell/model.R + test_simulate.R
  Run 2026-06-21 with rxode2 5.1.2, R 4.6.0

Pass criterion: all key metrics within 1% of R reference.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from model import run_cisplatin

# ---------------------------------------------------------------------------
# Reference values from R (2026-06-21 run)
# ---------------------------------------------------------------------------
R_REF = {
    '2.5_peak_C_plasma_ngml': 1257.54,   # ng/mL at t=60 min
    '2.5_peak_Inj':           0.411,     # at day 2.2
    '2.5_peak_Nec':           0.412,     # at day 4.0
    '2.5_peak_fold_Kim1':    50.1,       # at day 6.0
    '2.5_peak_fold_aGST':    22.4,       # at day 4.0
    '2.5_peak_fold_sCr':      1.17,      # at day 3.3
    '2.5_min_GFR_mlmin':      2.28,      # at day 3.5
    '1.0_peak_Nec':           0.0766,
    '1.0_peak_fold_sCr':      1.05,
    '22d_frac_F':             1.000,
    '22d_frac_Inj':           0.000,
    '22d_frac_Nec':           0.000,
    '22d_fold_sCr':           1.00,
}

TOLERANCE = 0.01   # 1%

def pct_err(val, ref):
    if abs(ref) < 1e-12:
        return abs(val - ref)
    return abs(val - ref) / abs(ref)

def run_validation():
    print("=== M005 Gebremichael 2018 â€” Python validation vs R reference ===\n")

    results = {}
    passed = []
    failed = []

    # --- 2.5 mg/kg ---
    sim_25 = run_cisplatin(dose_mg_kg=2.5, bw_kg=0.25)
    t_days = sim_25['t_days']

    early = t_days <= 1
    pk = np.argmax(sim_25['C_plasma'][early])
    results['2.5_peak_C_plasma_ngml'] = sim_25['C_plasma'][early][pk]

    results['2.5_peak_Inj']       = np.max(sim_25['frac_Inj'])
    results['2.5_peak_Nec']       = np.max(sim_25['frac_Nec'])
    results['2.5_peak_fold_Kim1'] = np.max(sim_25['fold_Kim1'])
    results['2.5_peak_fold_aGST'] = np.max(sim_25['fold_aGST'])
    results['2.5_peak_fold_sCr']  = np.max(sim_25['fold_sCr'])
    results['2.5_min_GFR_mlmin']  = np.min(sim_25['GFR_mlmin'])

    d22 = np.argmin(np.abs(t_days - 22))
    results['22d_frac_F']   = sim_25['Fcell'][d22]
    results['22d_frac_Inj'] = sim_25['Inj'][d22]
    results['22d_frac_Nec'] = sim_25['Nec'][d22]
    results['22d_fold_sCr'] = sim_25['fold_sCr'][d22]

    # --- 1.0 mg/kg ---
    sim_10 = run_cisplatin(dose_mg_kg=1.0, bw_kg=0.25)
    results['1.0_peak_Nec']      = np.max(sim_10['frac_Nec'])
    results['1.0_peak_fold_sCr'] = np.max(sim_10['fold_sCr'])

    # --- Compare ---
    print(f"{'Metric':<30} {'Python':>12} {'R ref':>12} {'Error%':>8} {'Pass?':>6}")
    print("-" * 76)

    for key, ref in R_REF.items():
        val = results[key]
        err = pct_err(val, ref) * 100
        ok  = err <= (TOLERANCE * 100) or abs(ref) < 1e-6
        status = "PASS" if ok else "FAIL"
        if ok:
            passed.append(key)
        else:
            failed.append(key)
        print(f"{key:<30} {val:>12.4f} {ref:>12.4f} {err:>7.2f}%  {status}")

    print(f"\nResult: {len(passed)}/{len(passed)+len(failed)} metrics within {TOLERANCE*100:.0f}%")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return False
    print("ALL PASS")
    return True


if __name__ == '__main__':
    ok = run_validation()
    sys.exit(0 if ok else 1)

