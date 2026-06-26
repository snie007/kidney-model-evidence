# M007 — Smith & Layton 2023 — Intrarenal RAS

| Field | Value |
|-------|-------|
| Paper | Smith D & Layton A, "The intrarenal renin-angiotensin system in hypertension: insights from mathematical modelling", J Math Biol 2023. PMID 36952058. |
| DOI   | 10.1007/s00285-023-01891-y |
| Modelling type | ODE; intrarenal RAS compartment model |
| Species | Rat |
| Software | Python (scipy, numpy) |
| Overall status | PASS |

## What the model does

Multi-compartment intrarenal angiotensin II model covering plasma, interstitium, tubular fluid, and cell compartments. Validated against published steady-state concentrations from Table III of the paper, achieving <1% error on all 9 AngII compartment concentrations.

## Validation summary

See REPLICATION_LOG.md for full status. Table III: 9/9 AngII compartment concentrations <1% error. PASS.

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_table3.py
```

## Dependencies
- numpy, scipy, matplotlib

## Key parameters
- ACE activity: angiotensin converting enzyme activity
- Renin secretion rate
- AT1R density per compartment

## Notes
Uses preprint (2021) parameter set, which differs from published (2023) paper by 3.1× for AngII_circ. Python port matches MATLAB reference code exactly.
