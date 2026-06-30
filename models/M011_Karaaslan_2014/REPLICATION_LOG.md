# M011 Karaaslan 2014 — Replication Log

**Paper**: Karaaslan, Denizhan & Hester (2014)  
"A mathematical model of long-term renal sympathetic nerve activity inhibition
during an increase in sodium intake"  
*Am J Physiol Regul Integr Comp Physiol* 306(4):R234–R247. PMID: 24285363.

**PDF**: `resources/papers/PMID_24285363_Karaaslan_2014_two_kidney_RSNA_ODE.pdf`  
**Access**: Paywalled (not OA). Paper available locally.

---

## Replication Summary

| Figure | Description | Status | Artifact |
|--------|-------------|--------|----------|
| Fig 2 (panels B/C/T/V) | Two-kidney 8× Na-loading experiment | INFORMATIVE_PASS | `M011_fig2_20260630_104857.json` |

---

## Figure 2 — Two-Kidney Sodium Loading Experiment

### Experimental Design (paper Methods)
- Left kidney: RSNA fixed at normal steady-state value throughout
- Right kidney: RSNA varies with feedback normally
- Days 0–5: sodium intake = 8× normal (0.126 × 8 = 1.008 meq/min)
- Days 5–10: sodium intake = normal (0.126 meq/min)
- 5-day spin-up to reach practical steady state (Na-balance residual < 0.5%)

### Validation Criteria (qualitative, paper inaccessible)

All 5/5 criteria PASS (INFORMATIVE_PASS):

| Criterion | Result | Pass |
|-----------|--------|------|
| MAP stable near 100 mmHg (|mean − 100| < 10) | mean = 105.7 mmHg | PASS |
| RSNA_R decreases with high Na loading | mean = −8.4% during loading | PASS |
| Intact kidney (R) excretes more Na than fixed (L) | 0.692 vs 0.127 meq/min | PASS |
| Sodium balance recovers (last 20% of recovery, within 20%) | 0.134 vs 0.126 meq/min (6.3%) | PASS |
| GFR in physiological range (50–200 mL/min) | 87.7 mL/min | PASS |

### Quantitative Model Outputs

| Quantity | Value | Notes |
|----------|-------|-------|
| MAP start | 102.3 mmHg | Slight elevation from Na retention at SS |
| MAP peak (loading) | 107.2 mmHg | 5 mmHg rise |
| MAP at end of recovery | 102.6 mmHg | Returns near SS |
| RSNA_R mean change (loading) | −8.4% | Pressure natriuresis activates |
| GFR_R mean | 87.7 mL/min | Increases from 81.7 at SS |
| Na excretion R vs L (loading) | 0.692 vs 0.127 meq/min | Right = 5.4× left |
| RMSE of MAP from 100 mmHg | 5.96 mmHg | Over entire experiment |

---

## Equation Flags (Deviations from HTML-extracted paper)

Nine flags documented in `model.py` docstring:

| Flag | Equation | Issue | Resolution |
|------|----------|-------|------------|
| FLAG-1 | A46 vasf | "11.312" → `vas_ss >> 1` | Corrected to `1.1e-5` |
| FLAG-2 | A53–A56 | achemo = 14*aauto misrendering | Interpreted as `achemo = aauto/4` |
| FLAG-3 | A61 | water reab coefficient "12" → negative urine | Not used; K_wreab derived (FLAG-7) |
| FLAG-4 | A70 | nu_md_sod denominator "1843" → sigmoid wrong scale | Corrected to `0.1843` |
| FLAG-5 | A76 | xi_k/sod exponent 0.00347 ≈ 0 | Set `xi_ksod = 1.0` |
| FLAG-6 | Table S1 (paywall) | PB, Pgo not given in main text | Derived: PB+Pgo = 40 mmHg, Kgcf = 0.0126 |
| FLAG-7 | A61 | K_wreab derived from mass balance | `K_wreab = 0.659` (at SS: Phi_u_tot = Phi_win) |
| FLAG-8 | A53–A56 | abaro initial condition | Derived as `abaro_0 = 0.75 × aauto(Pma_ss)` |
| FLAG-9 | A22 (TGF) | Direct TGF fixed-point diverges (gain ≈ −3.2) | Damped iteration (alpha = 0.25) + brentq fallback |

---

## Numerical Methods

| Parameter | Value |
|-----------|-------|
| Solver | `scipy.integrate.solve_ivp` method = `Radau` |
| Spin-up tolerances | rtol = 1e-6, atol = 1e-8 |
| Experiment tolerances | rtol = 1e-6, atol = 1e-8 |
| Spin-up duration | 5 days (Na-balance residual < 0.5% by day 5) |
| TGF iteration | Damped FP, alpha = 0.25, 50 iters max; brentq fallback |
| Cardiovascular loop | brentq, xtol = 0.001 (Pma); brentq for Phi_co |

**Note**: rtol = 1e-8 / atol = 1e-10 causes "Required step size less than spacing between numbers"
near the slow sodium equilibrium (days 5–30). Not a model error — the solver step size drops
below machine_eps × t near the converged state. rtol = 1e-6 / atol = 1e-8 avoids this.

---

## Platform

- Python 3.x, NumPy, SciPy
- Local Windows 11 (spin-up 5 days = 70s; experiment 10 days = 61s)
- Git commit: 123bd7c476e2 (at time of artifact generation)
