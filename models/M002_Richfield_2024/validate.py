# NOTE: run from the kidney-model-evidence repo root: python models/M002_Richfield_2024/validate.py
"""
validate.py â€” Validation of M002 Richfield 2024 Python port.

Runs the baseline (Pa=100 mmHg) and compares to R reference values.
Also runs the pressure-flow curve Pa=80..180 mmHg.

Strategy:
  - SNGFR, Q, Pgc_inlet, FF : from surrogate lookup (glom_SS table)
    Replicates R run_baseline.R which uses glom_SS().
  - Pgc_mean : from full capillary network solver (run_glom at 150 iters)
    Replicates R run_network.R which uses run_glom().

Exit code: 0 if all baseline checks pass within 1%, 1 otherwise.
"""

import sys
import numpy as np

from model import (
    run_baseline,           # full network solver (for Pgc_mean)
    run_baseline_surrogate, # surrogate lookup (for SNGFR/Q/FF/Pgc_inlet)
    params, IN_NODE, OUT_NODE,
)

# ============================================================
# Reference values
#   SNGFR, Pgc_inlet, FF  â€” from R run_baseline.R (surrogate lookup)
#   Pgc_mean              â€” from R run_network.R  (full solver)
# ============================================================
REF = {
    "SNGFR":     29.71,   # nl/min  (surrogate â†’ 29.724; full solver â†’ 29.71)
    "Pgc_mean":  49.88,   # mmHg   (full capillary network solver)
    "Pgc_inlet": 51.18,   # mmHg   (2*Pavg - Pa from surrogate)
    "FF":         0.292,  # dimensionless  (SNGFR / Q_plasma from surrogate)
}

TOL_PCT = 1.0  # 1% tolerance


def check_metric(name, value, ref, tol_pct=TOL_PCT):
    pct_err = abs(value - ref) / abs(ref) * 100.0
    passed = pct_err <= tol_pct
    status = "PASS" if passed else "FAIL"
    return status, pct_err, passed


def main():
    print("=" * 65)
    print("M002 Richfield 2024 â€” Validation vs R Reference")
    print("=" * 65)

    # ----------------------------------------------------------
    # Surrogate lookup: SNGFR, Pgc_inlet, Q, FF
    # (matches R run_baseline.R which uses glom_SS)
    # ----------------------------------------------------------
    print("\nRunning surrogate lookup (Pa=100 mmHg) ...")
    surr = run_baseline_surrogate(Pa=100.0)

    # ----------------------------------------------------------
    # Full network solver: Pgc_mean only
    # (matches R run_network.R which uses run_glom)
    # ----------------------------------------------------------
    print("Running full network solver for Pgc_mean (Pa=100 mmHg, 150 iters) ...")
    net = run_baseline(Pa=100.0, verbose=True)

    metrics = {
        "SNGFR":     surr["SNGFR"],
        "Pgc_mean":  net["Pgc_mean"],
        "Pgc_inlet": surr["Pgc_inlet"],
        "FF":        surr["FF"],
    }

    print("\n" + "-" * 65)
    print(f"{'Metric':<14} {'Python':>10} {'R ref':>10} {'% err':>8} {'Status':>8}")
    print("-" * 65)

    all_passed = True
    for name, ref_val in REF.items():
        py_val = metrics[name]
        status, pct_err, passed = check_metric(name, py_val, ref_val)
        print(f"{name:<14} {py_val:>10.4f} {ref_val:>10.4f} {pct_err:>7.3f}% {status:>8}")
        if not passed:
            all_passed = False

    print("-" * 65)
    print(f"\nSurrogate SNGFR:    {surr['SNGFR']:.4f} nl/min")
    print(f"Surrogate Q:        {surr['Q']:.4f} nl/min  (plasma flow)")
    print(f"Surrogate Pavg:     {surr['Pavg']:.4f} mmHg")
    print(f"Net-solver Pgc_mean:{net['Pgc_mean']:.4f} mmHg")
    print(f"Net-solver SNGFR:   {net['SNGFR']:.4f} nl/min  (150-iter non-converged)")
    print(f"Iterations:         {net['num_iter']}")

    # ----------------------------------------------------------
    # Pressure-flow curve Pa = 80..180 mmHg (surrogate)
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("Pressure-flow curve (Pa = 80 to 180 mmHg, step 10)")
    print("Using surrogate lookup table")
    print("=" * 65)
    print(f"{'Pa(mmHg)':<10} {'Q_pl(nl/min)':<14} {'SNGFR(nl/min)':<16} {'Pgc_inlet(mmHg)':<17} {'FF':<8}")
    print("-" * 65)

    Pa_range = np.arange(80, 185, 10)
    for Pa_try in Pa_range:
        try:
            r = run_baseline_surrogate(Pa=float(Pa_try))
            print(f"{Pa_try:<10.0f} {r['Q']:<14.4f} {r['SNGFR']:<16.4f} {r['Pgc_inlet']:<17.4f} {r['FF']:<8.4f}")
        except ValueError:
            # Outside surrogate range â€” fall back to full network solver
            try:
                r = run_baseline(Pa=float(Pa_try), verbose=False)
                in_mask = r['G']['Pa'] == float(Pa_try)  # approximate
                # Get Q_plasma from run_glom FF and SNGFR
                Q_pl = r['SNGFR'] / r['FF'] if r['FF'] > 0 else float('nan')
                print(f"{Pa_try:<10.0f} {Q_pl:<14.4f} {r['SNGFR']:<16.4f} {r['Pgc_inlet']:<17.4f} {r['FF']:<8.4f}  [network solver]")
            except Exception as ex2:
                print(f"{Pa_try:<10.0f} ERROR: {ex2}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    if all_passed:
        print("OVERALL: PASS â€” all baseline metrics within 1% of R reference")
        sys.exit(0)
    else:
        print("OVERALL: FAIL â€” one or more metrics exceed 1% tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()

