# M011 — Karaaslan, Denizhan & Hester 2014 — Two-Kidney RSNA Blood Pressure ODE

| Field | Value |
|-------|-------|
| Paper | Karaaslan F, Denizhan Y & Hester RL, "A mathematical model of long-term renal sympathetic nerve activity inhibition during an increase in sodium intake", Am J Physiol Regul Integr Comp Physiol 2014, 306(4):R234–R247. PMID 24285363. |
| DOI | 10.1152/ajpregu.00458.2013 |
| Modelling type | ODE; 10 state variables; Guyton-family long-term BP regulation |
| Species | Human |
| Software | Python (scipy, numpy) |
| Overall status | INFORMATIVE_PASS |

## What the model does

Two-kidney long-term blood pressure regulation ODE model (10 state variables, 40 functional blocks, 81 equations). Simulates the roles of the renal sympathetic nervous system (RSNA) in pressure natriuresis. Key experiment: fix RSNA in one kidney and allow normal feedback in the other to isolate the SNS contribution to sodium handling during an 8-fold sodium loading challenge.

Extends the Guyton model with explicit RSNA, tubuloglomerular feedback (TGF), myogenic autoregulation, and RAAS dynamics.

## Validation summary

See `REPLICATION_LOG.md` for full status.

| Figure | Status | Description |
|--------|--------|-------------|
| Fig 2 (B/C/T/V) | INFORMATIVE_PASS | 8× Na loading, 5-day load + 5-day recovery; 5/5 qualitative criteria met |

Note: Paper is paywalled. Validation against physiological targets and qualitative criteria from paper Methods (symmetric kidney design, RSNA sympathetic inhibition response).

## How to run

```bash
pip install -r ../../requirements.txt
python validate.py           # physiological SS check (PASS)
python replicate_fig2.py     # 8x Na loading experiment
```

## Dependencies

- numpy, scipy, matplotlib

## Numerical notes

- Solver: `scipy.integrate.solve_ivp`, method = `Radau` (stiff, 10 ODEs)
- Tolerances: rtol = 1e-6, atol = 1e-8 (spin-up and experiment)
- TGF loop requires damped fixed-point iteration (alpha = 0.25); direct substitution diverges (loop gain -3.2)
- Cardiovascular loop uses nested brentq solves (inner: Pma; outer: CO)
- 5-day spin-up sufficient for practical SS (Na-balance residual < 0.6% by day 5)

## Equation flags

Nine parameters/equations required correction from HTML-extracted values; full list in `model.py` docstring (FLAG-1 through FLAG-9). Key corrections:
- FLAG-2: autonomic multiplier epsilon_aum reinterpreted from HTML misrendering
- FLAG-4: nu_md_sod denominator 1843 -> 0.1843
- FLAG-6: PB + Pgo = 40 mmHg (not in main text; derived from filtration constraints)
- FLAG-7: K_wreab = 0.659 (derived from water mass balance; paper HTML gives unusable value)
- FLAG-9: TGF iteration requires damped scheme (paper implies direct substitution)
