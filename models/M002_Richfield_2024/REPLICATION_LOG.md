# M002 Replication Log — Richfield 2024 (PMID 38966231)

**Paper**: Richfield O, Bhatt DL, Feldman DI & others. "A multiscale model of renal
autoregulation and glomerular haemodynamics". *Frontiers in Physiology*, 2024.
DOI: 10.3389/fphys.2024.1336548. PMID: 38966231.

**PDF**: `resources/papers/PMID_38966231_Richfield_2024_glomcap_autoregulation.pdf`

---

## Figures

| Figure | Digitized CSV | Script | Last run | Metric | Status |
|--------|---------------|--------|----------|--------|--------|
| Fig 3A — Q_AA vs Pa (Pa=100 control only) | `M002_PMID38966231_fig3a_control.csv`, `…fig3a_furosemide.csv` | `replicate_fig3a.py` | 2026-06-25 | Pa=100: 1.37% error | **PASS** |

---

## Pass Criteria

### Figure 3A (primary) — Q_AA vs Perfusion Pressure
- Q_blood at Pa=100 (control condition, D_AA=7.0 µm) within **3%** of MATLAB reference
- PASS: 169.35 nl/min (Python) vs 167.06 nl/min (MATLAB reference) → 1.37% error ✓

### Validation (validate.py)
- SNGFR, Pgc_mean, Pgc_inlet, FF at Pa=100 within **1%** of R reference
- 4/4 PASS (errors 0.08–0.28%)

---

## Notes

### Python port scope
The Python port implements only the **glomerular capillary surrogate** (open-loop,
fixed D_AA). The full model includes:
- Myogenic response (D_AA adjusts with Pa)
- Tubuloglomerular feedback (TGF, D_AA responds to macula densa flow)

Validation of Figure 3A at Pa=125 and Pa=150 (which require autoregulation with
varying D_AA = 6.19, 5.73 µm from Takenaka 1994) is **out of scope** for the Python
port. The R code in `resources/code/autoreg_glommod/` handles the full autoregulation.

### Reference data source
The CSV reference values for Fig 3A are computed from the MATLAB autoregulation model
output file `Myo_TGF_model_curves_20231001.mat` (in `resources/code/autoreg_glommod/`),
using Q_blood = Q_plasma/(1-Ht) where Ht=0.4. They represent Q_AA at the equilibrium
D_AA values measured by Takenaka et al. 1994. Diltiazem condition omitted (Q_pred_dilt
is an empty array in the MATLAB output).

### Haematocrit
Richfield 2024 uses Ht=0.40. Q_blood = Q_plasma/(1-0.40).

---

## Artefact JSON files

| Run | File |
|-----|------|
| Fig3A 2026-06-25 | `artifacts/replication/M002_fig3a_20260625_163548.json` |
