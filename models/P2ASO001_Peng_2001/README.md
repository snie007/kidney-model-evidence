# P2ASO001 — Peng et al. 2001 — ASO Oligonucleotide PBPK

| Field | Value |
|-------|-------|
| Paper | Peng B et al., "Tissue Distribution and Physiologically Based Pharmacokinetics of Antisense Phosphorothioate Oligonucleotide ISIS 1082 in Rat", Antisense Nucleic Acid Drug Dev 2001. PMID 11219699. |
| DOI   | 10.1089/108729001750063608 |
| Modelling type | PBPK; whole-body 2-compartment permeability-limited model |
| Species | Rat |
| Software | Python (scipy, numpy) |
| Overall status | INFORMATIVE_PASS |

## What the model does

Whole-body PBPK model for phosphorothioate antisense oligonucleotide (ISIS 1082) distribution in rat, covering 14 tissues. Kidney and liver show highest uptake (Kp=38.8 and 12.7 respectively). Foundation model for ASO nephrotoxicity modelling in the MiMeC framework.

## Validation summary

See REPLICATION_LOG.md for full status. 5/6 tissues PASS (arterial blood, liver, intestine, kidney, muscle). Adipose tissue INFORMATIVE_FAIL (6× scatter in original paper data).

## How to run
```bash
pip install -r ../../requirements.txt
python replicate_fig2.py
```

## Dependencies
- numpy, scipy, matplotlib
- PyMuPDF (fitz) — only needed if re-digitizing from PDF; pre-computed CSVs are included

## Key parameters
- KI Kp=38.8 (kidney tissue-to-plasma partition coefficient)
- LI Kp=12.7 (liver tissue-to-plasma partition coefficient)
- KI fub·PS=4.69 mL/h (kidney permeability-surface area product)
- CLuR=2.11 mL/h (unbound renal clearance)

## Notes
Parameters taken directly from Tables 1 and 2 of the paper. Digitization of Figs 2A/2B/2C uses automated erosion-based square detection from PDF renders. Pre-computed CSVs are included so PyMuPDF is not needed to run the replication.
