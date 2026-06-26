# M002 — Richfield 2024 — Glomerular Autoregulation

| Field | Value |
|-------|-------|
| Paper | Richfield O et al., "A multiscale model of renal autoregulation and glomerular haemodynamics", Front Physiol 2024. PMID 38966231. |
| DOI   | 10.3389/fphys.2024.1336548 |
| Modelling type | Multi-scale ODE + surrogate (R → Python) |
| Species | Rat |
| Software | Python (scipy, numpy) |
| Overall status | PASS |

## What the model does

Models afferent arteriole resistance, glomerular capillary pressure, and single-nephron GFR. The Python port implements the glomerular capillary surrogate at fixed D_AA; full autoregulation requires the R code. Validated at Pa=100 mmHg against the MATLAB reference output showing 1.37% error.

## Validation summary

See REPLICATION_LOG.md for full status. Summary: Fig 3A at Pa=100 mmHg PASS (1.37% error). Pa=125 and Pa=150 require the R autoregulation loop and are out of scope.

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_fig3a.py
```

## Dependencies
- numpy, scipy, matplotlib
- pyreadr (for RDS surrogate file)

## Key parameters
- D_AA: afferent arteriole diameter, fixed at 7.0 µm (Takenaka control condition)
- Pgc: glomerular capillary pressure
- Pa: perfusion pressure (validated at 100 mmHg)
- HT: haematocrit (0.40)
- SNGFR: single-nephron GFR (reference 29.71 nl/min)

## Notes
Requires `data/surr_glom_df_20220724.RDS` (included). Full myogenic + TGF autoregulation requires the R code repository. The Python port covers only the glomerular capillary surrogate component.
