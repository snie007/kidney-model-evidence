# NOTE: run from the kidney-model-evidence repo root: python models/M003_Hallow_2017/validate.py
"""
Validation of the Python port of Hallow & Gebremichael (2017) renal QSP model.

Reference values from R/RxODE baseline run:
  MAP     = 93.11 mmHg
  GFR     = 94.34 mL/min

Additional physiological targets from the model parameterisation:
  RBF     â‰ˆ 1.0  L/min   (nom_renal_blood_flow_L_min)
  BV      â‰ˆ 5.0  L       (blood_volume_nom)
  Na_conc â‰ˆ 140  mEq/L   (ref_Na_concentration)
  serum_creatinine â‰ˆ 0.92 mg/dL (equilibrium_serum_creatinine)
  CO      â‰ˆ 5.0  L/min   (CO_nom)

Tolerance: â‰¤ 1 % vs reference for MAP and GFR (primary); â‰¤ 5% for others.
"""

import sys
from model import make_params, run_baseline


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------

REFS = {
    # Primary (confirmed R/RxODE output)
    "MAP":                   {"ref": 93.11,  "tol": 0.01, "primary": True,  "unit": "mmHg"},
    "GFR_ml_min":            {"ref": 94.34,  "tol": 0.01, "primary": True,  "unit": "mL/min"},
    # Secondary (nominal/parameterised targets)
    "renal_blood_flow_L_min":{"ref": 1.0,    "tol": 0.05, "primary": False, "unit": "L/min"},
    "blood_volume_L":        {"ref": 5.0,    "tol": 0.05, "primary": False, "unit": "L"},
    "Na_concentration":      {"ref": 140.0,  "tol": 0.05, "primary": False, "unit": "mEq/L"},
    "serum_creatinine_mg_dL":{"ref": 0.92,   "tol": 0.05, "primary": False, "unit": "mg/dL"},
    "cardiac_output":        {"ref": 5.0,    "tol": 0.05, "primary": False, "unit": "L/min"},
}


def main():
    print("=" * 72)
    print("Hallow 2017 Renal QSP â€” Python vs R/RxODE Validation")
    print("=" * 72)

    print("\nRunning baseline simulation (this may take ~1â€“2 minutes) â€¦\n")
    p = make_params()
    out = run_baseline(p)

    # --- Print results table ---
    header = f"{'Metric':<30} {'Python':>10} {'R ref':>10} {'% error':>9} {'Tol':>7} {'Status':>7}"
    print(header)
    print("-" * 72)

    all_pass     = True
    primary_pass = True

    for name, spec in REFS.items():
        ref  = spec["ref"]
        tol  = spec["tol"]
        prim = spec["primary"]
        unit = spec["unit"]

        py_val  = out.get(name, float("nan"))
        pct_err = abs(py_val - ref) / abs(ref) * 100.0
        ok      = pct_err <= tol * 100.0

        if not ok:
            all_pass = True and all_pass   # will be set False below
        if not ok and prim:
            primary_pass = False
        if not ok:
            all_pass = False

        flag   = "PASS" if ok else "FAIL"
        marker = " <-- PRIMARY" if prim else ""
        print(
            f"  {name:<28} {py_val:>10.4f} {ref:>10.4f} {pct_err:>8.3f}% "
            f"{tol*100:>6.1f}%  {flag}{marker}"
        )

    print("-" * 72)
    print()

    if all_pass:
        print("OVERALL RESULT: ALL CHECKS PASSED")
        sys.exit(0)
    elif primary_pass:
        print("OVERALL RESULT: PRIMARY CHECKS PASSED (some secondary checks failed)")
        sys.exit(0)
    else:
        print("OVERALL RESULT: FAILED â€” one or more PRIMARY checks did not meet tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()

