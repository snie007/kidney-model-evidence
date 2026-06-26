# M005 Replication Log — Gebremichael 2018 (PMID 29126144)

**Paper**: Gebremichael Y et al., "Quantitative systems toxicology model of acute kidney
injury and recovery following single-dose cisplatin exposure in rats."
*Toxicological Sciences*, 2018. PMID: 29126144.

**PDF**: `resources/papers/PMID_29126144_Gebremichael_2018_cisplatin_nephrotoxicity.pdf`

---

## Figures

| Figure | Digitized CSV | Script | Last run | RMSE / metric | Status |
|--------|---------------|--------|----------|--------------|--------|
| Fig 4A — Kim-1 2.5 mg/kg | M005_PMID29126144_fig4a_kim1.csv | replicate_fig4a.py | 2026-06-25 | 14.0% peak error | **PASS** |
| Fig 4A — aGST 2.5 mg/kg  | M005_PMID29126144_fig4a_agst.csv | replicate_fig4a.py | 2026-06-25 | 20.3% peak error | **PASS** |
| Fig 4A — sCr 2.5 mg/kg   | M005_PMID29126144_fig4a_scr.csv  | replicate_fig4a.py | 2026-06-25 | model 1.17x vs data 2.4x | EXCLUDED (see note) |
| Fig 4B — Kim-1 1.0 mg/kg | M005_PMID29126144_fig4b_kim1.csv | replicate_fig4b.py | 2026-06-25 | 40.6% peak error | **INFORMATIVE FAIL** |
| Fig 4B — aGST 1.0 mg/kg  | M005_PMID29126144_fig4b_agst.csv | replicate_fig4b.py | 2026-06-25 | 26.5% peak error | **PASS** |
| Fig 4B — sCr 1.0 mg/kg   | M005_PMID29126144_fig4b_scr.csv  | replicate_fig4b.py | 2026-06-25 | model 1.05x vs data ~1.1x | EXCLUDED (see note) |

---

## Pass Criterion

- **Threshold**: peak fold-change within 35% of Palmans et al. experimental data
- **Basis**: model was calibrated at 2.5 mg/kg; 1.0 mg/kg is a prediction test
- **sCr excluded**: model predicts 1.17× peak sCr vs ~2.4× data. Identical gap exists in
  R reference model. `secr0` and `S_TGF` are flagged ESTIMATED in `model.R` (tubular
  secretion and TGF gain not fitted to sCr data). Not a porting error.

---

## Notes

### sCr calibration gap (both doses)
- `secr0 = 0.1 mL/min` and `S_TGF = 0.20` are ESTIMATED parameters (not calibrated)
- Model correctly predicts direction (sCr rises, GFR falls after cisplatin)
- At S_TGF ≈ 0.67 the model would predict sCr ≈ 2.0× (matches data)
- This parameter pair is a candidate for AutoEmulate calibration

### Fig 4B Kim-1 INFORMATIVE FAIL (1.0 mg/kg)
- Model peak: 11.5×; data peak: 19.3× (40.6% error)
- Direction and timing are correct (peak day 6–7, returns to baseline by day 15)
- The model is calibrated at 2.5 mg/kg; the 1.0 mg/kg plot is a predictive test
- The paper's own simulation also underestimates at this dose level
- Not a porting error — the Python model reproduces the R model's behaviour

### Digitization methodology
- Source image: `resources/papers/M005_pages/page_07_hi.png` (page 7 of PDF, 300 dpi)
- Tool: automated PIL/numpy pixel extraction (`digitize_fig4.py`)
- Red hollow diamonds (Palmans data): R>165, G<130, B<130, R-G>60, min_cluster=80px
- Green triangles (Fukushima data): R<140, G>120, G-R>30, B<140, min_cluster=80px
- Panel A and B use different y-axis ranges (verified by visual inspection of tick labels):
  - Kim-1: Panel A y-max=80, Panel B y-max=30
  - aGST: Panel A y-max=40, Panel B y-max=5
  - sCr: Panel A y-max=3.0, Panel B y-max=1.5
- Column-3 panels (sCr, aGST): y-axis at x≈38 in crop (narrower left margin than column-1)

---

## Artefact JSON files

| Run | File |
|-----|------|
| Fig4A 2026-06-25 | `artifacts/replication/M005_fig4a_20260625_*.json` |
| Fig4B 2026-06-25 | `artifacts/replication/M005_fig4b_20260625_*.json` |
