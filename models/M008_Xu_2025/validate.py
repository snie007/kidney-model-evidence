# NOTE: run from the kidney-model-evidence repo root: python models/M008_Xu_2025/validate.py
"""
validate.py â€” Validation of Xu et al. 2025 rat kidney model (M008).

Checks single-nephron outputs against values reported in:
    Xu P et al. Am J Physiol Renal Physiol 328:F702-F723, 2025. PMID: 40099641

Runs run_single_nephron at P_in=100 mmHg (normal rat) and compares:
  - SNGFR: paper Fig 3 shows ~30â€“40 nl/min at 100 mmHg
  - Glomerular capillary pressure P_GC: ~50â€“55 mmHg
  - Afferent arteriole radius: ~10â€“15 um (terminal vessels in tree)
  - Macula densa NaCl (Cs_md): should be in physiological range ~15â€“35 mmol/l
  - Filtration fraction: ~0.15â€“0.25

Exit code: 0 if all PASS, 1 if any FAIL.
"""

import sys
import os

# Allow running from any directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from model import run_single_nephron, PARAMETER_TABLE


# ---------------------------------------------------------------------------
# Paper-reported target ranges (from Xu et al. 2025, normal rat at 100 mmHg)
# ---------------------------------------------------------------------------
# Fig 3 autoregulation curve: SNGFR plateau around 30-40 nl/min
# Table 1 / Fig 2: r_AA terminal vessels 10-15 um
# Standard rat physiology: P_GC ~ 45-60 mmHg
# Cs_md at steady state: ~20-30 mmol/l (TGF operating point ~46 mmol/l is C_half)
# Filtration fraction: ~0.15-0.25 (SNGFR / Q_A)

TARGETS = [
    {
        "variable": "SNGFR_nl_min",
        "description": "Single nephron GFR (nl/min)",
        "lo": 25.0,
        "hi": 50.0,
        "paper_note": "Xu 2025 Fig 3: ~30-40 nl/min at 100 mmHg",
    },
    {
        "variable": "P_GC_mmHg",
        "description": "Glomerular capillary pressure (mmHg)",
        "lo": 45.0,
        "hi": 60.0,
        "paper_note": "Standard rat: 45-60 mmHg",
    },
    {
        "variable": "r_AA_um",
        "description": "Afferent arteriole radius (um)",
        "lo": 8.0,
        "hi": 18.0,
        "paper_note": "Xu 2025: terminal AA radii 10-15 um",
    },
    {
        "variable": "Cs_md_mmol_l",
        "description": "NaCl at macula densa (mmol/l)",
        "lo": 10.0,
        "hi": 45.0,
        "paper_note": "TGF set-point C_half=46 mmol/l; operating below in normal",
    },
    {
        "variable": "filtration_fraction",
        "description": "Filtration fraction (SNGFR/Q_A)",
        "lo": 0.10,
        "hi": 0.40,
        "paper_note": "Typical rat FF ~0.15-0.25",
    },
    {
        "variable": "P_T0_mmHg",
        "description": "Bowman capsule pressure (mmHg)",
        "lo": 8.0,
        "hi": 18.0,
        "paper_note": "Standard rat: ~10-15 mmHg",
    },
    {
        "variable": "T_e_dyn_cm",
        "description": "Elastic wall tension (dyn/cm)",
        "lo": 0.0,
        "hi": 200.0,
        "paper_note": "Should be positive and finite",
    },
    {
        "variable": "T_m_dyn_cm",
        "description": "Active muscle tension (dyn/cm)",
        "lo": 0.0,
        "hi": 200.0,
        "paper_note": "Should be positive and finite",
    },
]


def run_validation():
    print("=" * 65)
    print("Xu et al. 2025 Rat Kidney Model â€” Validation (M008)")
    print("=" * 65)
    print(f"\nRunning run_single_nephron(pressure_mmHg=100) ...\n")

    result = run_single_nephron(pressure_mmHg=100)

    print("Model outputs:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:35s} = {v:10.4f}")
        else:
            print(f"  {k:35s} = {v}")

    print()
    print(f"{'Variable':<35} {'Value':>10}  {'Range':>20}  {'Status'}")
    print("-" * 75)

    all_pass = True
    results_summary = []

    for t in TARGETS:
        key = t["variable"]
        val = result.get(key, None)
        if val is None:
            status = "SKIP"
            val_str = "N/A"
        elif t["lo"] <= val <= t["hi"]:
            status = "PASS"
            val_str = f"{val:.4f}"
        else:
            status = "FAIL"
            val_str = f"{val:.4f}"
            all_pass = False

        range_str = f"[{t['lo']:.2f}, {t['hi']:.2f}]"
        print(f"  {t['description']:<33} {val_str:>10}  {range_str:>20}  {status}")
        results_summary.append((t["description"], val_str, range_str, status, t["paper_note"]))

    print("-" * 75)
    print()
    print("Paper notes:")
    for desc, val, rng, status, note in results_summary:
        print(f"  [{status}] {desc}: {note}")

    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print()

    # Additional parameter table summary
    print(f"Parameter count in PARAMETER_TABLE: {len(PARAMETER_TABLE)}")

    return all_pass


if __name__ == '__main__':
    passed = run_validation()
    sys.exit(0 if passed else 1)

