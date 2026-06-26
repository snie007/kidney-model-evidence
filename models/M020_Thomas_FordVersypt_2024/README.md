# M020 — Thomas & Ford Versypt 2024 — Macrophage/Fibrosis DKD

| Field | Value |
|-------|-------|
| Paper | Thomas SA & Ford Versypt ANF, "Mathematical Modeling of Macrophage Dynamics Dictating the Resolution of Glomerular Fibrosis in Diabetic Kidney Disease", Front Appl Math Stat 2024. PMID 39525640. |
| DOI   | 10.3389/fams.2024.1390413 |
| Modelling type | ODE QSP; macrophage-collagen DKD fibrosis model |
| Species | Rat |
| Software | Python (scipy, numpy) |
| Overall status | PASS (Fig 8B), INFORMATIVE_PASS (Fig 5B) |

## What the model does

Models macrophage recruitment, polarization, and collagen deposition in diabetic kidney disease glomeruli. Predicts resolution of fibrosis under different macrophage dynamics. Validated against digitized figure data from the original paper.

## Validation summary

See REPLICATION_LOG.md for full status. Fig 8B (collagen): PASS (28%/6% error). Fig 5B (macrophage): INFORMATIVE_PASS due to sequential fitting artefact.

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_fig5b.py
python replicate_fig8b.py
```

## Dependencies
- numpy, scipy, matplotlib

## Key parameters
- mu_COL: collagen deposition rate
- k_Mf: M1 macrophage activation rate
- k_r: M2 repair rate

## Notes
Sequential fitting artefact documented: MAC fold-change INFORMATIVE_PASS because MAC parameters were fitted first, then COL parameters fitted holding MAC fixed.
