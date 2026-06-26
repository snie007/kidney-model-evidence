# Kidney Mechanistic Model Evidence Pack

Python implementations of 7 published kidney mechanistic models with automated replication evidence. Each model directory contains: the Python ODE implementation, digitized reference data from the original paper, replication comparison figures, and timestamped JSON artifacts recording pass/fail outcomes.

## Model summary

| Model | Paper | Year | Modelling type | Topic | Figures validated | Status |
|-------|-------|------|----------------|-------|-------------------|--------|
| M002 | Richfield et al. | 2024 | Multi-scale ODE | Glomerular autoregulation | Fig 3A | PASS |
| M003 | Hallow & Gebremichael | 2017 | ODE QSP | Blood pressure / Na regulation | 6 figures | PASS |
| M005 | Gebremichael et al. | 2018 | ODE QST | Cisplatin nephrotoxicity | Fig 4A–4B | PASS / INFORMATIVE |
| M007 | Smith & Layton | 2023 | ODE | Intrarenal RAS | Table III | PASS |
| M008 | Xu et al. | 2025 | ODE | Renal hemodynamics | Fig 6 | PASS |
| M020 | Thomas & Ford Versypt | 2024 | ODE QSP | Macrophage/fibrosis DKD | Fig 5B, 8B | PASS / INFORMATIVE |
| P2ASO001 | Peng et al. | 2001 | PBPK | ASO oligonucleotide distribution | Fig 2A–2C | INFORMATIVE_PASS |

## Replication evidence standard

Every model has:
- **Digitized CSV data** from paper figures (automated extraction where applicable)
- **A replication script** (`replicate_*.py`) that runs the model and compares to reference data
- **A timestamped JSON artifact** in `evidence/` recording input CSV filenames, model output values, RMSE, and pass/fail status
- **A PNG comparison figure** in `evidence/` showing model vs digitized reference data

## How to run

```bash
pip install -r requirements.txt

# Run individual models
python models/M002_Richfield_2024/replicate_fig3a.py
python models/M003_Hallow_2017/replicate_desc_fig2.py
python models/M003_Hallow_2017/replicate_app_fig1.py
python models/M003_Hallow_2017/replicate_app_fig2.py
python models/M003_Hallow_2017/replicate_app_fig3.py
python models/M005_Gebremichael_2018/replicate_fig4a.py
python models/M005_Gebremichael_2018/replicate_fig4b.py
python models/M007_Smith_Layton_2023/replicate_table3.py
python models/M008_Xu_2025/replicate_fig6.py
python models/M020_Thomas_FordVersypt_2024/replicate_fig5b.py
python models/M020_Thomas_FordVersypt_2024/replicate_fig8b.py
python models/P2ASO001_Peng_2001/replicate_fig2.py
```

## Repository structure

```
kidney-model-evidence/
├── README.md
├── requirements.txt
└── models/
    ├── M002_Richfield_2024/
    │   ├── README.md
    │   ├── model.py
    │   ├── validate.py
    │   ├── replicate_fig3a.py
    │   ├── REPLICATION_LOG.md
    │   ├── data/           <- digitized CSVs + RDS surrogate
    │   └── evidence/       <- PNG + JSON artifacts
    ├── M003_Hallow_2017/
    │   ├── README.md
    │   ├── model.py
    │   ├── validate.py
    │   ├── replicate_desc_fig2.py
    │   ├── replicate_app_fig1.py
    │   ├── replicate_app_fig2.py
    │   ├── replicate_app_fig3.py
    │   ├── REPLICATION_LOG.md
    │   ├── data/           <- digitized CSVs
    │   └── evidence/       <- PNGs + JSON artifacts
    ├── M005_Gebremichael_2018/
    │   ├── README.md
    │   ├── model.py
    │   ├── validate.py
    │   ├── replicate_fig4a.py
    │   ├── replicate_fig4b.py
    │   ├── REPLICATION_LOG.md
    │   ├── data/           <- digitized CSVs
    │   └── evidence/       <- PNGs + JSON artifacts
    ├── M007_Smith_Layton_2023/
    │   ├── README.md
    │   ├── model.py
    │   ├── validate.py
    │   ├── replicate_table3.py
    │   ├── REPLICATION_LOG.md
    │   ├── data/           <- digitized CSV
    │   └── evidence/       <- PNG + JSON artifact
    ├── M008_Xu_2025/
    │   ├── README.md
    │   ├── model.py
    │   ├── validate.py
    │   ├── replicate_fig6.py
    │   ├── REPLICATION_LOG.md
    │   ├── data/           <- empty (model-only validation)
    │   └── evidence/       <- PNG + JSON artifact
    ├── M020_Thomas_FordVersypt_2024/
    │   ├── README.md
    │   ├── model.py
    │   ├── validate.py
    │   ├── replicate_fig5b.py
    │   ├── replicate_fig8b.py
    │   ├── REPLICATION_LOG.md
    │   ├── data/           <- digitized CSVs
    │   └── evidence/       <- PNGs + JSON artifacts
    └── P2ASO001_Peng_2001/
        ├── README.md
        ├── model.py
        ├── replicate_fig2.py
        ├── REPLICATION_LOG.md
        ├── data/           <- digitized CSVs
        └── evidence/       <- PNG + JSON artifact
```
