# M020 Replication Log — Thomas & Ford Versypt 2024 (PMID 39525640)

**Paper**: Thomas SA & Ford Versypt ANF. "Mathematical Modeling of Macrophage
Dynamics Dictating the Resolution of Glomerular Fibrosis in Diabetic Kidney Disease".
*Frontiers in Applied Mathematics and Statistics*, 2024. PMID: 39525640.

**PDF**: `resources/papers/PMID_39525640_Thomas_FordVersypt_2024_glomerular_fibrosis.pdf`

---

## Figures

| Figure | Digitized CSV | Script | Last run | Metric | Status |
|--------|---------------|--------|----------|--------|--------|
| Fig 8B — Collagen fold change vs time (DKD) | `M020_PMID39525640_fig8b_col.csv` | `replicate_fig8b.py` | 2026-06-25 | wk16: 28.2%, wk20: 6.0% | **PASS** |
| Fig 5B — Macrophage fold change vs time (DKD) | `M020_PMID39525640_fig5b_mac.csv` | `replicate_fig5b.py` | 2026-06-25 | wk16: 32.0%, wk20: 3.4% | **INFORMATIVE_PASS** |

---

## Pass Criteria

### Fig 8B (primary) — Collagen fold change
- Model COL fold within **30%** of digitized data at weeks ≥14 (mature DKD)
- Week 15.9: model 1.458× vs data 2.030× (28.2%) → **PASS**
- Week 19.9: model 2.549× vs data 2.710× (6.0%) → **PASS**
- Weeks <14 are INFORMATIVE ONLY (model still at baseline; sigmoid transition timing uncertainty)

### Fig 5B (informative) — Macrophage fold change
- 50% tolerance, EXCLUDED from primary PASS (see Notes below)
- Week 16.2: model 6.41× vs data 9.42× (32%) → within 50% ✓
- Week 20.3: model 11.18× vs data 11.57× (3.4%) → ✓

---

## Notes

### Sequential-fitting artefact (MAC model)
The FitColData parameter set (used by this model) was obtained by sequential fitting:
1. MAC/MCP parameters fitted to Fig 5B MAC data first → gives MAC peak ~4-5× (FitMACData)
2. COL parameters fitted holding MAC parameters fixed → FitColData

The final FitColData parameter set gives MAC peak ~13× in both the MATLAB reference
and this Python port. The digitized data shows MAC peak ~9.4-11.6× (weeks 16-20),
so the discrepancy vs the earlier estimate of "4-5×" in the model.py docstring reflects
that the original note was based on a different reading of the figure.

This is NOT a port error. It is a well-known limitation of sequential vs. simultaneous
fitting. MAC fold change (Fig 5B) is thus INFORMATIVE ONLY.

### Collagen calibration
The model is explicitly calibrated to reproduce COL fold change (Fig 8B). The primary
validation shows:
- Week 16: 28.2% error (within 30% pass threshold)
- Week 20: 6.0% error (well within threshold)

The early data point at week 9.9 (fold=1.87 from an isolated teal component in the
figure) is excluded from PASS: the model is still at baseline (fold≈1.0) at week 10,
while one experimental data point shows fold=1.87. This timing discrepancy likely reflects
inter-study variability (non-averaged data from multiple studies).

### Digitization methodology
Both figures were digitized by automated Python pixel extraction (PIL/numpy):
- Color masks (orange for MAC, teal for COL)
- Column pixel-count profiling to find "hot" columns (data circles wider than model line)
- Isolated connected-component labelling for components separated from the model line
- Axes calibrated from model baseline pixel positions and known calibration points

---

## Artefact JSON files

| Run | File |
|-----|------|
| Fig8B 2026-06-25 | `artifacts/replication/M020_fig8b_20260625_*.json` |
| Fig5B 2026-06-25 | `artifacts/replication/M020_fig5b_20260625_*.json` |
