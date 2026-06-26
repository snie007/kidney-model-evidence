# M003 — Hallow & Gebremichael 2017 — Blood Pressure / Na Regulation

| Field | Value |
|-------|-------|
| Paper | Hallow KM & Gebremichael Y, "A quantitative systems physiology model of renal function and blood pressure regulation: Model description", CPT Pharmacometrics Syst Pharmacol 2017. PMID 28548387. Application paper PMID 28556624. |
| DOI   | 10.1002/psp4.12178 / 10.1002/psp4.12177 |
| Modelling type | ODE QSP; 40+ state blood pressure/Na homeostasis model |
| Species | Human |
| Software | Python (scipy, numpy) |
| Overall status | PASS |

## What the model does

Integrates renal Na handling, RAAS, cardiac output, and pressure natriuresis into a whole-body blood pressure regulation model. Validated on salt-sensitive vs salt-resistant hypertension across 6 figures from the description paper (PMID 28548387) and application paper (PMID 28556624).

## Validation summary

See REPLICATION_LOG.md for full status. All 6 figures PASS.

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_desc_fig2.py
python replicate_app_fig1.py
python replicate_app_fig2.py
python replicate_app_fig3.py
```

## Dependencies
- numpy, scipy, matplotlib

## Key parameters
- SP-N: salt-sensitivity slope
- baseline_nephrons: number of nephrons
- nominal_ACE_activity: ACE enzyme activity
- nom_Kf: nominal filtration coefficient
- nominal_map_setpoint: mean arterial pressure setpoint (100.16 mmHg for application paper)

## Notes
Calibration uses 2D root-find for application paper (nominal_map_setpoint=100.16, nom_Kf=2.38). Description paper uses default parameters.
