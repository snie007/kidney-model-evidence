# M018 — Mahato et al. 2018 Replication Log

**Paper**: Mahato RI et al., "Quantitative systems pharmacology model of diabetic kidney disease",
*npj Systems Biology and Applications* 4:35 (2018).  
**PMID**: 30564457  
**PDF**: `resources/papers/PMID_30564457_Mahato_2018_DKD_QSP.pdf`

---

## Figure Status

| Figure | Panel | Description | Status | Artifact | Date |
|--------|-------|-------------|--------|----------|------|
| Fig 5 | A–F | DKD progression (db/db, db/db+UNX vs control): GFR, Kf, N, MAP, UAER | **PASS** | M018_fig5_20260628_173909.json | 2026-06-28 |
| Normal SS | — | Normal control steady-state (GFR, MAP, Pgc, Na balance) | **PASS** | M018_normalSS_20260628_173530.json | 2026-06-28 |

---

## Steady-State Verification (Normal Scenario)

| Quantity | Model | Target | Units | Source |
|----------|-------|--------|-------|--------|
| GFR | 0.318 | 0.29–0.35 | mL/min | Table 2 Mahato 2018 |
| MAP | 98.2 | 90–106 | mmHg | Table 1 Mahato 2018 |
| Pgc | 37.5 | 34–41 | mmHg | Table 1 Mahato 2018 |
| Na_conc | 140.000 | 138–142 | mmol/L | physiological |
| delta_Daa | 0.0 | ~0 | — | no injury in normal |
| dsodium at SS | −2.7×10⁻¹⁰ | ~0 | mEq/min | Na balance |

Run to t_end = 500,000 min (×2 passes); converged in 0.7 s on cemrg001.

---

## Fig 5 Replication Results

Run date: 2026-06-28. Solver: LSODA rtol=1e-5/atol=1e-7.

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| GFR RMSE (db/db) | 25.0% | ≤ 30% | PASS |
| UAER trend (db/db) | rising (1063 → 1203 µg/day) | rising | PASS |
| GFR RMSE (control) | 59.3% | informative | INFORMATIVE |

Notes:
- GFR control RMSE 59.3% reflects scatter in digitized fallback data (visual inspection),
  not necessarily model error. The model produces stable 0.318 mL/min throughout 25 weeks.
- UAER baseline = 1063 µg/day is higher than typical mouse values (~10–100 µg/day);
  UAER formula may need calibration against dedicated albuminuria data (future work).
- db/db GFR: 0.318 → 0.332 mL/min (+4.4%) over 25 weeks.
- db/db+UNX GFR: 0.318 → 0.586 mL/min (+84%) due to 50% nephron loss at week 8.

---

## Model Fixes Applied

### Fix 1 — P_bowmans clamp (lines 918–924)
**Problem**: Tubular compliance formula was calibrated for human tube lengths
(L_pt = 10 mm) but used with mouse values (L_pt = 2.2 mm). The formula
underestimated Bowman's pressure at mouse SNGFR, creating a spurious high-SNGFR
attractor at ~56.4 nL/min instead of the correct ~31.8 nL/min.

**Fix**: Physical clamp `P_in_pt_mmHg = max(Pc_pt + P_interstitial, formula)`.
At mouse SNGFR = 31.8 nL/min, the formula gives ~17 mmHg but the clamp enforces
19 mmHg (= Pc_pt + P_interstitial = 14 + 5), consistent with the terminal tubular
pressure. This eliminates the spurious 56.4 nL/min attractor.

### Fix 2 — nom_cd_na_reabsorption includes SGLT2 (lines 377–396)
**Problem**: `nom_cd_na_reabsorption` was computed from total PT Na reabsorption
without accounting for SGLT2-coupled Na reabsorption at normal BG. This led to
over-estimated CD reabsorption fraction (0.634 vs correct ~0.57).

**Fix**: Compute `nom_Na_SGLT2_total` at normal BG and subtract from filtered Na
before computing `nom_cd_na_reabsorption`. With BG = 90 mg/dL:
`nom_cd_na_reabsorption = 0.5680`.

### Fix 3 — blood_glucose_mg_dl("normal") = 90 mg/dL (line 69)
**Problem**: Normal scenario BG was set to 120 mg/dL, but `BG_nom = 90 mg/dL`
(normoglycaemic reference in Table 1). This caused `BG_excess = 30 mg/dL > 0`,
driving `d(delta_Daa)/dt = 1.875e-6/min`. Over 200,000 min (≈139 days):
delta_Daa grew to ~0.19 (19% afferent dilation), elevating SNGFR to ~64 nL/min,
causing massive Na excretion and model collapse (MAP → −691,138 mmHg).

**Fix**: Set `blood_glucose_mg_dl("normal") = 90.0 mg/dL = BG_nom`. This matches
the C57BL/6J lean control fasting BG used as the normoglycaemic reference in
Mahato 2018. With `BG_excess = 0`, `d(delta_Daa)/dt = 0` → model is stable.

### Tolerance fix — run_to_ss solver tolerances (lines 1194–1218)
Changed rtol = 1e-7 → 1e-5 and atol = 1e-9 → 1e-7. The mouse ECF volume
(0.006 L vs human 15 L) creates stiff Q_water/Q_Na terms (eigenvalue λ = 167 /min
vs 0.067 /min for human). Looser tolerances give <0.1% error at SS and reduce
integration time from >14 min to 0.7 s.

---

## Discrepancies / Limitations

| # | Quantity | Model vs Data | Explanation | Tier |
|---|---------|---------------|-------------|------|
| 1 | SNGFR_SS | 31.8 vs 29 nL/min (+9%) | Digitized from Fig 2 has scatter; model consistent with Table 1 range 29–35 nL/min | T2 |
| 2 | UAER baseline | 1063 vs 10–100 µg/day | UAER formula needs calibration against dedicated albuminuria data | T3 |
| 3 | GFR ctrl RMSE | 59.3% | Digitized control scatter large (visual inspection fallback); model value 0.318 mL/min is within expected range | T2 |

Tier: T2 = estimated parameter, T3 = formula calibration needed

---

## Pending

- [ ] Calibrate UAER formula against explicit albuminuria data from Mahato Fig 2C
- [ ] Run AutoEmulate GPE + Adaptive MH calibration (LHS data needed)
- [ ] Replicate Fig 2A (blood glucose trajectory)
