# M007 Replication Log — Smith & Layton 2023 (PMID 36952058)

**Paper**: Smith D & Layton A. "The intrarenal renin-angiotensin system in hypertension:
insights from mathematical modelling". *Journal of Mathematical Biology*, 2023.
DOI: 10.1007/s00285-023-01891-y. PMID: 36952058.

**Preprint**: bioRxiv 2021. DOI: 10.1101/2021.12.14.472639.
**Preprint PDF**: `resources/papers/S010_Smith_Layton_2023_intrarenal_RAS_preprint.pdf`
**Published PDF**: Not downloaded — J Math Biol (Springer) is paywalled.

---

## Figures

| Figure | Digitized CSV | Script | Last run | Metric | Status |
|--------|---------------|--------|----------|--------|--------|
| Table III — SS AngII in all compartments | `M007_PMID36952058_table3_ss.csv` | `replicate_table3.py` | 2026-06-25 | 9/9 AngII vars <1% | **PASS** |

---

## Pass Criteria

### Table III (primary) — Steady-state concentrations
- AngII in all 9 compartments within **1%** of preprint Table III values
- All 9 PASS (errors 0.005–0.175%)

### Validation against MATLAB reference
- `validate.py` compares Python port to MATLAB reference (MATLAB_REF_7DAY)
- 35/35 variables within 1% — all PASS (errors ≈ 0.000%)

---

## Notes

### Preprint vs published parameter discrepancy
The MATLAB code (and Python port) use the **preprint** (2021) parameter set:
- AngII_circ: **43.4 fmol/mL** (preprint / MATLAB / Python)
- AngII_circ: **14 fmol/mL** (published paper, 2023, different fitted params)

This is a documented 3.1× discrepancy between the code repository and the published paper.
The MATLAB code was not updated to match the published parameter set. The Python port
correctly reproduces the MATLAB code output.

The `replicate_table3.py` validates against the **preprint Table III** values (matching
MATLAB), which are the correct target for the Python port. The note is documented in the
JSON artifact and in `model.py`.

### Key references
- GitHub repository: https://github.com/Layton-Lab/intrarenalRAS
- Local code: `resources/code/intrarenalRAS/`
- Python port: `resources/python/M007_Smith_Layton_2023/model.py`

---

## Artefact JSON files

| Run | File |
|-----|------|
| Table3 2026-06-25 | `artifacts/replication/M007_table3_20260625_171104.json` |
