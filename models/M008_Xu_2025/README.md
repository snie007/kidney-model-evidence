# M008 — Xu et al. 2025 — Renal Hemodynamics

| Field | Value |
|-------|-------|
| Paper | Xu P et al., "Full-scale model of renal hemodynamics using vascular tree", Am J Physiol Renal Physiol 2025. PMID 40099641. |
| DOI   | 10.1152/ajprenal.00309.2024 |
| Modelling type | ODE; single-nephron myogenic autoregulation |
| Species | Rat |
| Software | Python (scipy, numpy) |
| Overall status | PASS |

## What the model does

Models afferent arteriole myogenic response and TGF to regulate single-nephron GFR. Validated against 5 hemodynamic targets from Table 4 of the paper at Pa=100 mmHg. Full vascular tree requires VTK data not included here.

## Validation summary

See REPLICATION_LOG.md for full status. Fig 6 (Table 4 targets): 5/5 hemodynamic targets PASS at Pa=100 mmHg.

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_fig6.py
```

## Dependencies
- numpy, scipy, matplotlib

## Key parameters
- r_AA: afferent arteriole radius
- P_GC: glomerular capillary pressure
- Cs_md: macula densa NaCl concentration (corrected 20.6→45.3 mM)

## Notes
Full autoregulation plateau (Fig 10 RBF) requires VTK vascular tree files not available. Figs 3B and 10 are PENDING. The data/ directory is empty as M008 uses model-only (Table 4) validation.
