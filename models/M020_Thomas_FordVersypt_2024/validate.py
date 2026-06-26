# NOTE: run from the kidney-model-evidence repo root: python models/M020_Thomas_FordVersypt_2024/validate.py
"""
Validation of M020 Python port against MATLAB reference values
(Thomas & Ford Versypt 2024, Glomerular Fibrosis Model)

MATLAB reference obtained by running runGlomerularFibrosis.m with
ModelFitResultsFinal.mat on 2026-06-21.

Tolerance:
  Primary:   <=1%  relative error vs MATLAB reference
  Secondary: <=5%  relative error vs paper (fold-change data)
"""

import sys
import numpy as np
sys.path.insert(0, ".")
from model import run_fibrosis, STATE_NAMES

# ---------------------------------------------------------------------------
# MATLAB reference values (from run_reference.m output, 2026-06-21)
# Columns: AGE, MCP, MAC, TGF, AMC, MMP, TIMP, COL
# ---------------------------------------------------------------------------
MATLAB_REF = {
    # week: (day, [AGE, MCP, MAC, TGF, AMC, MMP, TIMP, COL])
    0:  ( 1, [1.49942e-07, 1.78346e-10, 3.36728e-13, 9.90895e-10,
               6.35328e-04, 1.45726e-06, 9.04501e-10, 2.94319]),
    4:  (28, [1.49942e-07, 1.78346e-10, 3.36728e-13, 9.90895e-10,
               6.35328e-04, 1.45726e-06, 9.04501e-10, 2.94319]),
    8:  (56, [1.77503e-07, 1.81004e-10, 3.48106e-13, 9.95469e-10,
               6.36764e-04, 1.50404e-06, 9.06381e-10, 2.9432]),
    12: (84, [5.87004e-07, 2.12703e-10, 6.87179e-13, 1.13184e-09,
               7.02864e-04, 2.94478e-06, 9.19982e-10, 3.02239]),
    16: (112,[1.98758e-06, 2.59647e-10, 2.08003e-12, 1.69210e-09,
               1.31734e-03, 8.92169e-06, 9.23479e-10, 4.33495]),
    20: (140,[3.87929e-06, 2.81988e-10, 3.69360e-12, 2.34124e-09,
               2.75989e-03, 1.59229e-05, 9.19763e-10, 7.52946]),
    24: (168,[5.36202e-06, 2.90337e-10, 4.49945e-12, 2.66544e-09,
               3.52005e-03, 1.94251e-05, 9.18639e-10, 8.85549]),
}

# Paper fold-change reference (vs week 8) â€” from MATLAB model output
# These are the model-fitted curve values (i.e. the paper's fitted simulation),
# not raw experimental data points. The model fits an ensemble of studies;
# individual study data points scatter above/below.
# Values are taken from the MATLAB reference run (ModelFitResultsFinal.mat).
# COL fold-change relative to week-8 COL (COL_week8 = 2.9432):
PAPER_REF_COL_FC_VS_W8 = {
    8:  1.0000,  # Reference
    16: 1.4729,  # MATLAB model (fitted curve) â€” see Fig 2 of paper
    20: 2.5583,  # MATLAB model (fitted curve) â€” see Fig 2 of paper
}

TOLERANCE_MATLAB = 0.01   # 1%
TOLERANCE_PAPER  = 0.05   # 5%

# ---------------------------------------------------------------------------
# Run Python model
# ---------------------------------------------------------------------------
print("Running Python DKD base case...")
t, y, y_ss = run_fibrosis(scenario="DKD", simulation_weeks=24)
print(f"  Healthy SS: COL={y_ss[7]:.6g}  MAC={y_ss[2]:.6g}\n")


# ---------------------------------------------------------------------------
# Helper: interpolate model at a specific day
# ---------------------------------------------------------------------------
def interp_at_day(t_arr, y_arr, day):
    idx = np.searchsorted(t_arr, day)
    if idx >= len(t_arr):
        idx = len(t_arr) - 1
    return y_arr[idx]


# ---------------------------------------------------------------------------
# Table 1: All-state comparison vs MATLAB reference
# ---------------------------------------------------------------------------
print("=" * 90)
print("TABLE 1 â€” All states vs MATLAB reference (tolerance: 1%)")
print("=" * 90)
header = f"{'Wk':>3}  {'Var':>5}  {'Python':>14}  {'MATLAB':>14}  {'Err%':>8}  {'Pass?':>6}"
print(header)
print("-" * 90)

all_pass_matlab = True
week_col_python = {}

for week in sorted(MATLAB_REF.keys()):
    day, ref_vals = MATLAB_REF[week]
    py_vals = interp_at_day(t, y, day)

    for i, name in enumerate(STATE_NAMES):
        ref = ref_vals[i]
        py  = py_vals[i]
        if abs(ref) < 1e-30:
            rel_err = abs(py - ref)
            err_str = f"{rel_err:.2e} (abs)"
            pass_flag = rel_err < 1e-20
        else:
            rel_err_pct = abs(py - ref) / abs(ref) * 100.0
            err_str = f"{rel_err_pct:.4f}%"
            pass_flag = rel_err_pct <= TOLERANCE_MATLAB * 100

        marker = "PASS" if pass_flag else "FAIL"
        if not pass_flag:
            all_pass_matlab = False
        print(f"{week:>3}  {name:>5}  {py:>14.6g}  {ref:>14.6g}  {err_str:>8}  {marker:>6}")

    week_col_python[week] = py_vals[7]  # save COL for fold-change table

print("-" * 90)
overall_matlab = "ALL PASS" if all_pass_matlab else "SOME FAIL"
print(f"MATLAB comparison: {overall_matlab}\n")


# ---------------------------------------------------------------------------
# Table 2: Collagen fold-change vs MATLAB reference (key output)
# ---------------------------------------------------------------------------
print("=" * 70)
print("TABLE 2 â€” Collagen (COL) fold-change vs MATLAB (tolerance: 1%)")
print("=" * 70)
header2 = f"{'Week':>6}  {'COL_py':>12}  {'COL_mat':>12}  {'Err%':>8}  {'FC_py':>7}  {'FC_mat':>7}  {'Pass?':>6}"
print(header2)
print("-" * 70)

col0_matlab = MATLAB_REF[0][1][7]
col0_python = week_col_python[0]
all_pass_col = True

for week in sorted(MATLAB_REF.keys()):
    col_mat = MATLAB_REF[week][1][7]
    col_py  = week_col_python[week]
    rel_err_pct = abs(col_py - col_mat) / abs(col_mat) * 100.0
    fc_py  = col_py  / col0_python
    fc_mat = col_mat / col0_matlab
    pass_flag = rel_err_pct <= TOLERANCE_MATLAB * 100
    if not pass_flag:
        all_pass_col = False
    marker = "PASS" if pass_flag else "FAIL"
    print(f"{week:>6}  {col_py:>12.6g}  {col_mat:>12.6g}  {rel_err_pct:>7.4f}%  {fc_py:>7.4f}  {fc_mat:>7.4f}  {marker:>6}")

print("-" * 70)
overall_col = "ALL PASS" if all_pass_col else "SOME FAIL"
print(f"Collagen vs MATLAB: {overall_col}\n")


# ---------------------------------------------------------------------------
# Table 3: COL fold-change vs paper Fig 2 data (tolerance 5%)
# (Paper normalises to week 8 value, so we compute COL_FC vs week 8)
# ---------------------------------------------------------------------------
print("=" * 70)
print("TABLE 3 â€” COL fold-change vs paper (Fig 2) reference (tolerance: 5%)")
print("  Note: paper FCs are relative to week-8 COL")
print("=" * 70)
header3 = f"{'Week':>6}  {'FC_py_8':>9}  {'FC_paper':>9}  {'Err%':>8}  {'Pass?':>6}"
print(header3)
print("-" * 70)

col_week8_py = float(week_col_python[8])
all_pass_paper = True

for week, fc_paper in sorted(PAPER_REF_COL_FC_VS_W8.items()):
    day = MATLAB_REF[week][0]
    col_py = float(interp_at_day(t, y, day)[7])
    fc_py_vs_8 = col_py / float(col_week8_py)

    rel_err_pct = abs(fc_py_vs_8 - fc_paper) / abs(fc_paper) * 100.0
    pass_flag = bool(rel_err_pct <= TOLERANCE_PAPER * 100)
    if not pass_flag:
        all_pass_paper = False
    marker = "PASS" if pass_flag else "FAIL"
    print(f"{week:>6}  {fc_py_vs_8:>9.4f}  {fc_paper:>9.4f}  {rel_err_pct:>7.4f}%  {marker:>6}")

print("-" * 70)
overall_paper = "ALL PASS" if all_pass_paper else "SOME FAIL"
print(f"Paper comparison: {overall_paper}\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"  vs MATLAB all states:    {overall_matlab}")
print(f"  vs MATLAB collagen:      {overall_col}")
print(f"  vs paper fold-changes:   {overall_paper}")
print()

all_ok = all_pass_matlab and all_pass_col and all_pass_paper
if all_ok:
    print("OVERALL: PASS")
    sys.exit(0)
else:
    print("OVERALL: FAIL")
    sys.exit(1)

