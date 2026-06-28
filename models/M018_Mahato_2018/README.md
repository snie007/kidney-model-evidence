# M018 — Mahato et al. 2018 — Diabetic Kidney Disease QSP

| Field | Value |
|-------|-------|
| Paper | Mahato RI et al., "Quantitative systems pharmacology model of diabetic kidney disease", npj Syst Biol Appl 4:35 (2018). PMID 30564457. |
| DOI   | 10.1038/s41540-018-0070-3 |
| Modelling type | ODE QSP — extends M003 (Hallow 2017) with SGLT2 glucose transport, adaptive Kf hypertrophy, and nephron loss |
| Species | Mouse (C57BL/6J; db/db diabetic model) |
| Software | Python (scipy, numpy) |
| Overall status | PASS |

## What the model does

Extends the Hallow 2017 renal QSP (M003) with three DKD-specific mechanisms:

1. **SGLT2 glucose reabsorption**: Michaelis-Menten tubular glucose transport with SGLT2 and SGLT1 components; couples blood glucose elevation to increased Na reabsorption and glomerular hyperfiltration.
2. **Adaptive Kf hypertrophy**: Slow ODE (τ ≈ 750 days) tracking glomerular basement membrane expansion driven by TGF-β/glucose exposure.
3. **Nephron loss**: ODE coupling progressive glomerulosclerosis to functional nephron count reduction.

Mouse parameterisation: ECF = 0.006 L, L_pt = 2.2 mm, BG_nom = 90 mg/dL (C57BL/6J fasting BG), SNGFR_SS ≈ 31.8 nL/min.

## Validation summary

| Figure | Description | Status |
|--------|-------------|--------|
| Normal SS | GFR, MAP, Pgc, Na balance at baseline | PASS |
| Fig 5 A–F | 25-week DKD progression (db/db, db/db+UNX vs control) | PASS |

GFR RMSE (db/db vs digitized data) = 25.0% ≤ 30% threshold. UAER trend correctly rising in db/db.

See REPLICATION_LOG.md for full details including 3 critical bug fixes applied during porting.

## How to run

```bash
pip install -r ../../requirements.txt
python replicate_fig5.py
```

## Dependencies

- numpy, scipy, matplotlib

## Key parameters

- `BG_nom`: normoglycaemic reference BG = 90 mg/dL (must match `blood_glucose_mg_dl("normal")`)
- `N_nephrons_nom`: nominal nephron count (mouse: 16,000)
- `Kf_nom`: baseline filtration coefficient (nL/min/mmHg)
- `k_Daa`: afferent arteriole dilation rate constant driven by hyperglycaemia
- `tau_Kf_days`: Kf adaptation time constant (~750 days)

## Scenarios

| Scenario key | Description |
|---|---|
| `"normal"` | Lean C57BL/6J control (BG = 90 mg/dL) |
| `"dbdb"` | db/db diabetic (BG = 300 mg/dL, hyperglycaemia from week 4) |
| `"dbdb_unx"` | db/db + uninephrectomy at week 8 (50% nephron mass reduction) |

## Notes

- The model is parameterised for mouse, not human. Human baseline would require different ECF, L_pt, and tubular transport parameters.
- UAER baseline (≈1063 µg/day) is higher than typical mouse values (10–100 µg/day); the UAER formula may require calibration against dedicated albuminuria data (future work).
- Solver uses LSODA with rtol=1e-5, atol=1e-7 to handle the stiff mouse ECF ODEs (eigenvalue λ ≈ 167 /min).
