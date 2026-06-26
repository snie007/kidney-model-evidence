# P2ASO001 Replication Log — Peng 2001 (PMID 11219699)

**Paper**: Peng B, Andrews J, Nestorov I, Brennan B, Nicklin P, Rowland M.
"Tissue Distribution and Physiologically Based Pharmacokinetics of Antisense
Phosphorothioate Oligonucleotide ISIS 1082 in Rat."
*Antisense Nucleic Acid Drug Dev.* 2001;11(1):15-27. PMID: 11219699.

**PDF**: `resources/papers/P2-ASO-001_Peng_2001_ASO_ISIS1082_PBPK_rat.pdf`

---

## Figures

| Figure | Tissue | Digitized CSV | Script | Last run | log10-RMSE | Status |
|--------|--------|---------------|--------|----------|------------|--------|
| Fig 2A | ART (arterial blood) | `P2ASO001_PMID11219699_fig2a_art.csv` | `replicate_fig2.py` | 2026-06-26 | 0.136 | **PASS** |
| Fig 2A | LI (liver)           | `P2ASO001_PMID11219699_fig2a_li.csv`  | `replicate_fig2.py` | 2026-06-26 | 0.141 | **PASS** |
| Fig 2B | IN (intestine)       | `P2ASO001_PMID11219699_fig2b_in.csv`  | `replicate_fig2.py` | 2026-06-26 | 0.088 | **PASS** |
| Fig 2B | KI (kidney)          | `P2ASO001_PMID11219699_fig2b_ki.csv`  | `replicate_fig2.py` | 2026-06-26 | 0.210 | **PASS** |
| Fig 2C | MU (muscle)          | `P2ASO001_PMID11219699_fig2c_mu.csv`  | `replicate_fig2.py` | 2026-06-26 | 0.168 | **PASS** |
| Fig 2C | AD (adipose)         | `P2ASO001_PMID11219699_fig2c_ad.csv`  | `replicate_fig2.py` | 2026-06-26 | 0.296 | **INFORMATIVE_FAIL** |

**Overall: INFORMATIVE_PASS** (5 PASS, 1 INFORMATIVE_FAIL)

---

## Pass Criteria

- **Primary**: log10-RMSE < 0.25 for each tissue (geometric mean factor-of-1.78 error)
- **Adipose exception**: up to 0.35 log10 acceptable — adipose data has 6× inter-animal scatter
  at single time points (e.g., C=4.2 vs C=25.3 µg/g at t≈0.2h from individual animals)

### Cross-validation against paper text values

| Tissue | t (h) | Digitized (µg/g) | Paper text (µg/g) | Error |
|--------|--------|------------------|-------------------|-------|
| ART    | 24     | 3.62             | 3.6 ± 0.1         | 0.5%  |
| LI     | 24     | 56.4             | 57 ± 7.7          | 1.0%  |
| KI     | 24     | 174.7            | 184 ± 27          | 5.1%  |

---

## Notes

### Model scope

Python implementation of the Peng 2001 two-compartment permeability-limited PBPK:
- 14 tissues + arterial + venous blood = 30 ODE states
- Parameters from Tables 1 (physiological) and 2 (fitted) of the paper
- IV bolus initialised in venous blood: C_VEN(0) = dose_µg / V_VEN_mL
- Integrator: LSODA (scipy solve_ivp), rtol=1e-8, atol=1e-10

### Digitization method

Automated PIL/scipy erosion detection from PDF (3× zoom render):
- 5×5 morphological erosion removes thin model-fit lines; filled square markers survive
- Surviving components with post-erosion size ≥ 50 pixels classified as data points
- Panel axis calibration derived empirically: x_left=col339, x_right=col1347 (from
  cross-referencing known t=24h and t=72h data point positions across panels)
- y-axis per panel: log-scale bounds read from rendered axis tick labels

### Adipose INFORMATIVE_FAIL

The adipose tissue has the highest inter-animal variability in the paper. The figure
(Fig 2C, bottom panel) shows data spanning:
- t≈0.2h: C = 4.2, 9.1, 25.3 µg/g (6× spread from n=3 animals)
- t≈24h:  C = 1.5, 2.0, 6.9 µg/g (4.5× spread)

The PBPK model predicts the geometric mean; RMSE=0.296 log10 units (factor 1.98×)
is consistent with the observed biological scatter. This is a data limitation,
not a model deficiency.

---

## Artifact JSON files

| Run | File |
|-----|------|
| Fig2 2026-06-26 | `artifacts/replication/P2ASO001_fig2_20260626_073852.json` |
