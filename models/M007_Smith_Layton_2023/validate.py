# NOTE: run from the kidney-model-evidence repo root: python models/M007_Smith_Layton_2023/validate.py
"""
validate.py â€” Smith & Layton 2023 Intrarenal RAS Model
Compares the Python port against MATLAB reference values.

Reference: MATLAB run_baseline_check.m, 7-day baseline, no Ang II infusion.
Tolerance: <=1% relative error vs MATLAB reference.

Exit code: 0 if all pass, 1 if any fail.
"""

import sys
import numpy as np

from model import run_baseline, get_params, MATLAB_REF_7DAY, STATE_NAMES

# ---------------------------------------------------------------------------
# MATLAB reference values (7-day baseline, from run_baseline_check.m)
# ---------------------------------------------------------------------------
REF = MATLAB_REF_7DAY

# Variables to validate (only differential-state variables, not algebraic zeros)
VALIDATE_VARS = [
    'AGT_circ',
    'AngI_circ',
    'AngII_circ',
    'AT1R_AngII_memb_circ',
    'Ang17_circ',
    'PRC',
    'PRA',
    'AngI_Isf_Gl',
    'AngII_Isf_Gl',
    'AT1R_AngII_memb_Gl',
    'AT1R_AngII_cell_Gl',
    'AngII_cell_Gl',
    'AT1R_memb_Gl',
    'AT1R_cell_Gl',
    'AngI_Isf_Pt',
    'AngII_Isf_Pt',
    'AT1R_AngII_memb_Pt',
    'AT1R_AngII_cell_Pt',
    'AngII_cell_Pt',
    'AT1R_memb_Pt',
    'AT1R_cell_Pt',
    'AngI_Fl_Tb',
    'AngII_Fl_Tb',
    'AT1R_AngII_memb_Tb',
    'AT1R_AngII_cell_Tb',
    'AngII_cell_Tb',
    'AT1R_memb_Tb',
    'AT1R_cell_Tb',
    'AngI_Pv',
    'AngII_Pv',
    'AT1R_AngII_memb_Pv',
    'AT1R_memb_Pv',
    'AngI_T',
    'AngII_T',
    'nu_AT1R',
]

TOLERANCE = 0.01  # 1%


def run_validation():
    print("=" * 72)
    print("Smith & Layton 2023 â€” Intrarenal RAS Model: Python vs MATLAB")
    print("=" * 72)
    print(f"Simulation: 7 days, no Ang II infusion")
    print(f"Tolerance:  {TOLERANCE*100:.1f}%")
    print()

    # Run Python model
    print("Running Python model (LSODA, rtol=1e-8, atol=1e-10) ...")
    p = get_params()
    result, sol = run_baseline(p, days=7, rtol=1e-8, atol=1e-10)
    print(f"Solver status: {sol.message}")
    print(f"Steps taken:   {len(sol.t)}")
    print()

    # Print results table
    header = f"{'Variable':<30} {'MATLAB':>15} {'Python':>15} {'RelErr%':>9} {'Status':>6}"
    print(header)
    print("-" * len(header))

    n_pass = 0
    n_fail = 0
    failures = []

    for var in VALIDATE_VARS:
        ref_val = REF[var]
        py_val  = result[var]

        if abs(ref_val) < 1e-12:
            # Near-zero reference: check absolute error
            abs_err = abs(py_val - ref_val)
            rel_err_pct = abs_err * 100  # use absolute as %-of-zero is undefined
            pass_check = abs_err < 1e-6
        else:
            rel_err = abs(py_val - ref_val) / abs(ref_val)
            rel_err_pct = rel_err * 100
            pass_check = rel_err <= TOLERANCE

        status = "PASS" if pass_check else "FAIL"
        if pass_check:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((var, ref_val, py_val, rel_err_pct))

        flag = "" if pass_check else " <=="
        print(f"{var:<30} {ref_val:>15.6g} {py_val:>15.6g} {rel_err_pct:>8.4f}% {status:>6}{flag}")

    print("-" * len(header))
    print(f"\nResults: {n_pass} PASS, {n_fail} FAIL out of {len(VALIDATE_VARS)} variables")
    print()

    if n_fail == 0:
        print("OVERALL: PASS")
        print()
        print("All Python port values match MATLAB reference within 1%.")
        print("Validation: MATLAB-equivalent (run from model_SS.mat initial conditions,")
        print("7-day baseline simulation, LSODA solver).")
        return 0
    else:
        print("OVERALL: FAIL")
        print()
        print("Failing variables:")
        for var, ref, py, err in failures:
            print(f"  {var}: MATLAB={ref:.6g}, Python={py:.6g}, err={err:.4f}%")
        return 1


if __name__ == '__main__':
    sys.exit(run_validation())

