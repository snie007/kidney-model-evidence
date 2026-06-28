# M006 Replication Log — Maass 2019 (PMID 30869201)

**Paper**: Maass C, Sorensen NB, Himmelfarb J, Kelly EJ, Stokes CL, Cirit M.
"Translational Assessment of Drug-Induced Proximal Tubule Injury Using a Kidney
Microphysiological System."
*CPT Pharmacometrics Syst Pharmacol.* 2019;8(5):316-325.
DOI: 10.1002/psp4.12400

**Local PDF**: `resources/papers/S009_Maass_2019_kidney_MPS_QSP.pdf`

## Model Summary

QSP/PBPK model for in vitro-to-in vivo translation (IVIVT) of kidney injury
biomarker (KIM-1) from a kidney microphysiological system (kidney MPS / organ-on-chip)
to human plasma and urine levels.

**Three modules:**
1. Whole-body PBPK for cisplatin (and rifampicin, gentamicin) — standard
   multi-compartment ODE model, physiological parameters from Brown 1997 [ref 34]
   and ICRP Valentin 2002 [ref 33].
2. KIM-1 shedding rate: empirically derived from in vitro MPS measurements
   (Figure 3), scaled to N_nephrons = 1.8×10⁶ (Scotcher 2016 [ref 35]).
3. Neutrophil recruitment: piecewise time-course of neutrophil fold-changes
   (Awad 2009 [ref 36]) × KIM-1 boost factor 3.25 (Lingadahalli 2013 [ref 37]).

**NOTE**: The exact ODE equations appear in Supplementary Methods S2 of the
Wiley publication. These are not reproduced in the main-paper PDF and could not
be accessed programmatically (journal paywall + PoW download challenge).
This implementation reconstructs the model from the main-paper description.

## Figures

| Figure | Type | Digitized CSV | Script | Last run | Status |
|--------|------|---------------|--------|----------|--------|
| Fig 3a | Experimental (cisplatin KIM-1 in MPS) | `resources/digitized/M006_PMID30869201_fig3a_cisplatin.csv` | `digitize_fig3.py` | 2026-06-28 | DATA ONLY (visual_inspection_fallback) |
| Fig 5 (a,b,c) | Computational (plasma/urine KIM-1 simulation) | N/A (vs paper text targets) | `replicate_fig5.py` | 2026-06-28 | INFORMATIVE_PASS |

## Latest Results (2026-06-28)

Artifact: `artifacts/replication/M006_fig5_20260628_153846.json`

| Criterion | Target | Model Output | Pass? |
|-----------|--------|-------------|-------|
| Plasma KIM-1 peak (no immune) | ≥ 2× baseline (≥100 pg/mL) | 105.2 pg/mL | ✓ |
| Plasma KIM-1 peak (with immune) | 300–3000 pg/mL | 474.6 pg/mL | ✓ |
| Urine KIM-1 peak (with immune) | 1000–10000 pg/mL | 3141.9 pg/mL | ✓ |
| Peak timing (plasma, immune) | 8–48 h | 32.2 h | ✓ |

**Overall status: INFORMATIVE_PASS (4/4 criteria)**

## Figure Categories

- **Figure 2**: MPS hardware + KIM-1 2D vs MPS — experimental only, no model output to replicate.
- **Figure 3**: KIM-1 in MPS during drug exposure — experimental input data; used to parameterise shedding model.
- **Figure 4**: Metabolic activity (PrestoBlue) — experimental only.
- **Figure 5**: **Computational target** — simulated plasma/urine KIM-1 profiles with/without immune effect.
- **Figure 6**: Dosing regimen optimization — computational output, requires full rifampicin PBPK (out of scope).

## Pass Criteria (Figure 5)

Per paper main text (PMID 30869201 p. 321):
1. Plasma KIM-1 peak (no immune): ≥ 2× baseline (~100 pg/mL)
2. Plasma KIM-1 peak (with immune): ~1000 pg/mL (range 300–3000 pg/mL accepted)
3. Urine  KIM-1 peak (with immune): ~3000 pg/mL (range 1000–10000 pg/mL accepted)
4. Peak timing: 12–36 h post-dose

Note: Wider acceptance range used because exact supplementary parameters (S2)
are inaccessible; this is a model reconstruction, not a code verification.
Therefore INFORMATIVE_PASS is the maximum achievable status.

## How to Run

```bash
# Step 1: Digitize Figure 3a
ssh cemrg001 "tmux send-keys -t fleet:kidney 'cd /home/sneidere/kidney-mechanistic && gate python resources/python/M006_Maass_2019/digitize_fig3.py' Enter"

# Step 2: Replicate Figure 5
ssh cemrg001 "tmux send-keys -t fleet:kidney 'cd /home/sneidere/kidney-mechanistic && gate python resources/python/M006_Maass_2019/replicate_fig5.py' Enter"

# Copy results
scp cemrg001:/home/sneidere/kidney-mechanistic/artifacts/replication/M006_fig5_*.json artifacts/replication/
scp cemrg001:/home/sneidere/kidney-mechanistic/figures/replication/M006_fig5.png figures/replication/
```

## Evidence Trail

- Raw figure images: `resources/digitized/raw_images/PMID_30869201_fig3.png`
- Fig 3a CSV: `resources/digitized/M006_PMID30869201_fig3a_cisplatin.csv`
- Replication artifacts: `artifacts/replication/M006_fig5_<timestamp>.json`
- Replication figure: `figures/replication/M006_fig5.png`

## Limitations / Known Gaps

1. **Supplementary S2 inaccessible**: Exact ODEs and drug-specific parameters
   are in Methods S2 (not in main paper). Model parameters are reconstructed
   from main-text description + literature PBPK defaults.
2. **Figure 3 drug concentrations**: Exact MPS drug concentrations (Table S2)
   not published in main text; values estimated from same MPS system (Adler 2016).
3. **Figure 6** (dosing optimization for rifampicin): Not replicated — requires
   rifampicin PBPK + Emax model + dosing optimization; out of scope for initial port.
4. **Virtual patient population**: N=100 virtual patients with 20-25% parameter
   variability; exact variability distributions from supplementary.
5. **Single drug**: Only cisplatin implemented. Rifampicin and gentamicin
   would require separate PBPK parameterization.
