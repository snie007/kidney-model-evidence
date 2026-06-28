"""
Validate M018 Mahato 2018 Python port — normal steady-state check.

Reference values from Table 1/2 of Mahato et al. 2018 (PMID 30564457)
and model verification run 2026-06-28 on cemrg001.

Run from repo root: python models/M018_Mahato_2018/validate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import make_params, run_to_ss, compute_outputs

REFS = {
    "GFR_mL_min":     {"ref": 0.318,  "tol": 0.05, "primary": True,  "unit": "mL/min",   "source": "Table 2 Mahato 2018 (0.29-0.35)"},
    "MAP_mmHg":       {"ref": 98.2,   "tol": 0.05, "primary": True,  "unit": "mmHg",     "source": "Table 1 Mahato 2018 (90-106)"},
    "Pgc_mmHg":       {"ref": 37.5,   "tol": 0.05, "primary": True,  "unit": "mmHg",     "source": "Table 1 Mahato 2018 (34-41)"},
    "SNGFR_nL_min":   {"ref": 31.8,   "tol": 0.10, "primary": False, "unit": "nL/min",   "source": "Table 2 Mahato 2018 (29-35)"},
    "Na_conc_mM":     {"ref": 140.0,  "tol": 0.02, "primary": False, "unit": "mmol/L",   "source": "physiological"},
}


def main():
    print("=" * 70)
    print("M018 Mahato 2018 DKD QSP — Normal Steady-State Validation")
    print("=" * 70)

    print("\nRunning lean control steady-state (normal BG = 90 mg/dL) ...\n")
    p = make_params("normal")
    _, y_ss = run_to_ss(p)
    out = compute_outputs(y_ss, p)

    header = f"{'Metric':<28} {'Model':>10} {'Ref':>10} {'%err':>8} {'Tol':>6} {'Status':>7}"
    print(header)
    print("-" * 70)

    all_pass = True
    primary_pass = True

    for name, spec in REFS.items():
        ref  = spec["ref"]
        tol  = spec["tol"]
        prim = spec["primary"]
        unit = spec["unit"]
        src  = spec["source"]

        val = out.get(name, float("nan"))
        pct_err = abs(val - ref) / abs(ref) * 100.0
        ok = pct_err <= tol * 100.0

        if not ok:
            all_pass = False
            if prim:
                primary_pass = False

        flag = "PASS" if ok else "FAIL"
        marker = " <-- PRIMARY" if prim else ""
        print(f"  {name:<26} {val:>10.4f} {ref:>10.4f} {pct_err:>7.2f}% {tol*100:>5.0f}%  {flag}{marker}")

    print("-" * 70)

    if all_pass:
        print("\nOVERALL: ALL CHECKS PASSED")
        sys.exit(0)
    elif primary_pass:
        print("\nOVERALL: PRIMARY CHECKS PASSED (some secondary checks failed)")
        sys.exit(0)
    else:
        print("\nOVERALL: FAILED — one or more PRIMARY checks did not meet tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
