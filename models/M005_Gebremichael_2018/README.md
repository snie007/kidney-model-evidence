# M005 — Gebremichael et al. 2018 — Cisplatin Nephrotoxicity

| Field | Value |
|-------|-------|
| Paper | Gebremichael Y et al., "Quantitative systems toxicology model of acute kidney injury and recovery following single-dose cisplatin exposure in rats", Toxicol Sci 2018. PMID 29126144. |
| DOI   | 10.1093/toxsci/kfx254 |
| Modelling type | ODE QST; cisplatin-induced tubular injury |
| Species | Rat |
| Software | Python (scipy, numpy) |
| Overall status | PASS (Fig 4A), INFORMATIVE (Fig 4B) |

## What the model does

Models cisplatin-induced proximal tubule cell death and recovery, validated on urinary biomarkers Kim-1 and aGST and serum creatinine. Calibrated at 2.5 mg/kg; 1.0 mg/kg is a predictive test dose.

## Validation summary

See REPLICATION_LOG.md for full status. Fig 4A at 2.5 mg/kg: Kim-1 PASS (14%), aGST PASS (20%). Fig 4B at 1.0 mg/kg: Kim-1 INFORMATIVE FAIL (40.6%, known single-dose calibration limitation). sCr excluded from validation.

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_fig4a.py
python replicate_fig4b.py
```

## Dependencies
- numpy, scipy, matplotlib

## Key parameters
- Kcat: cisplatin cytotoxicity rate
- Kdie: cell death rate
- Kreg: tubular cell regeneration rate

## Notes
sCr excluded from validation: secr0 and S_TGF are estimated, not calibrated. Kim-1 at 1.0 mg/kg INFORMATIVE FAIL is a known limitation of single-dose calibration.
