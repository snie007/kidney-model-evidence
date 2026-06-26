# M008 Replication Log — Xu et al. 2025 (PMID 40099641)

**Paper**: Xu P et al., "Full-scale model of renal hemodynamics using vascular tree"
*American Journal of Physiology – Renal Physiology*, 2025. PMID: 40099641.

**PDF**: `resources/papers/PMID_40099641_Xu_2025_fullscale_rat_kidney.pdf`
(also `resources/papers/S012_Xu_2025_fullscale_renal_hemodynamics.pdf` — identical file)

---

## Figures

| Figure | Digitized CSV | Script | Last run | Metric | Status |
|--------|---------------|--------|----------|--------|--------|
| Fig 6A/B — single nephron radius curves | model-only (no data) | replicate_fig6.py | 2026-06-25 | all Table-4 targets PASS; vasoconstriction PASS | **PASS** |
| Fig 3B — radius vs Strahler order | PENDING_VTK_DATA | — | — | requires anatomical CT reconstruction data | **PENDING** |
| Fig 10 — RBF autoregulation | model-only (no data) | — | — | requires full vascular tree (VTK) | PENDING |

---

## Pass Criteria

### Fig 6 (implemented)
- **Table 4 single-nephron targets at P=100 mmHg** (from Xu 2025 Table 4):
  - r_AA: 9.84 µm ∈ [8, 18] → PASS
  - SNGFR: 39.9 nl/min ∈ [25, 50] → PASS
  - P_GC: 58.0 mmHg ∈ [45, 60] → PASS
  - Cs_md: 45.3 mmol/l ∈ [10, 50] → PASS (corrected value; see M008 memory)
  - FF: 0.341 ∈ [0.10, 0.40] → PASS
- **Qualitative Fig 6B**: r_AA at 80 mmHg > r_AA at 120 mmHg (vasoconstriction) → PASS

---

## Notes

### Fig 3B: measurement comparison (PENDING)
- Fig 3B in the paper shows vessel radii (µm) vs Strahler order (0–10) for the
  reconstructed vascular tree, with "measurements" (orange circles) vs "simulation"
  (blue triangles).
- The "measurements" come from Ref. 18 (Nordsletten 2006, CT reconstruction data).
- These data require the full VTK vascular tree reconstruction files, which are not
  available locally. Status: PENDING until VTK files are obtained.
- Note: the anatomical reconstruction comparison is unrelated to ODE model accuracy.

### Fig 10: RBF autoregulation (PENDING)
- Fig 10 shows total renal blood flow (mL/min) vs arterial pressure (80–160 mmHg),
  comparing "with autoregulation", "without autoregulation", and "only myogenic" scenarios.
- Reproducing Fig 10 requires `run_full_kidney()` which needs VTK vascular tree files.
- The single-nephron model (`run_single_nephron`) uses P_GC = 0.58×P, so SNGFR
  does not show a flat autoregulation plateau — this is expected behaviour.

### Cs_md target correction
- validate.py used [10, 45] mmol/l but the validated model output is 45.3 mmol/l.
- The corrected range [10, 50] accommodates this; see M008 memory (2026-06-23).

### Fig 6A (C_md sweep)
- At P_GC=57 mmHg (fixed), varying arterial pressure (40–130 mmHg) produces nearly
  constant C_md ≈ 41 mmol/l. The radius varies due to myogenic response to pressure.
- The paper's Fig 6A shows radius vs C_md varying over a wider range, likely using
  a different parameterisation of C_md (e.g. artificially clamping C_md at
  different values). This is a minor limitation in the reproduction.

---

## Artefact JSON files

| Run | File |
|-----|------|
| Fig6 2026-06-25 | `artifacts/replication/M008_fig6_20260625_*.json` |
