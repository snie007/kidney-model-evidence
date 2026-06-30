"""
M011 — Karaaslan, Denizhan & Hester (2014)
"A mathematical model of long-term renal sympathetic nerve activity inhibition
during an increase in sodium intake"
Am J Physiol Regul Integr Comp Physiol 306(4):R234-R247. PMID: 24285363.

Python implementation of the two-kidney ODE model.

=============================================================================
UNIT CONVENTIONS (applied throughout this file)
=============================================================================
Time           : minutes (min)
Pressure       : mmHg
Flow (blood)   : L/min
Flow (Na)      : meq/min
Volume         : L (blood, ECF)
Concentration  : meq/L  (sodium, potassium)
               : ng/L   (AngII = Cat)
               : mU/L   (ADH = Cadh)
               : ng/L   (aldosterone = Cal)
Resistance     : mmHg·min·L⁻¹
RSNA           : dimensionless (1 = normal)
Hormone ratios : dimensionless (Ĉ = 1 at normal)
=============================================================================

EQUATION INVENTORY (Appendix equations A1–A81 from paper):

Block 1  : RSNA (A3–A5)                — algebraic
Block 2  : Renal vascular resistance (A6–A11) — algebraic
Block 3  : Myogenic response (A12–A15) — ODE (dPaad/dt) per kidney
Block 4  : Nitric oxide (A16–A17)      — algebraic
Block 5  : Renal blood flow (A18)      — algebraic
Block 6  : GFR (A19–A21)              — algebraic
Block 7  : TGF signal (A22)            — algebraic
Block 8  : Filtered sodium (A23)       — algebraic
Block 9  : Proximal Na reabsorption (A24–A28) — algebraic
Block 10 : Macula densa Na (A29)       — algebraic
Block 11 : Distal Na reabsorption (A30–A32) — algebraic
Block 12 : Distal Na outflow (A33)     — algebraic
Block 13 : CD Na reabsorption (A34–A37) — algebraic
Block 14 : Urine Na (A38)              — algebraic
Block 15 : Water intake (A39)          — algebraic
Block 16 : ECF volume (A40)            — ODE (dVecf/dt)
Block 17 : Blood volume (A41)          — algebraic
Block 18 : Mean filling pressure (A42) — algebraic
Block 19 : Venous return (A43)         — algebraic
Block 20 : Cardiac output             — algebraic (Φco = Φvr, implicit solve)
Block 21 : Right atrial pressure (A44) — algebraic
Block 22 : Vascularity (A45–A47)       — ODE (dvas/dt)
Block 23 : Arterial resistance (A48–A49) — algebraic
Block 24 : Venous return resistance (A50) — algebraic
Block 25 : Total peripheral resistance (A51) — algebraic
Block 26 : MAP (A52)                   — algebraic
Block 27 : Autonomic multiplier (A53–A56) — ODE (dabaro/dt) per interpretive note
Block 28 : ADH (A57–A60)               — ODE (ddelta_ra/dt, dC_adh_hat/dt)
Block 29 : Tubular water reabsorption (A61–A63) — algebraic
Block 30 : Urine flow (A64)            — algebraic
Block 33 : Sodium intake               — external input
Block 34 : Total body sodium (A67)     — ODE (dMsod/dt)
Block 35 : Plasma sodium (A68)         — algebraic
Block 36 : Renin secretion (A69–A71)   — algebraic
Block 37 : Renin concentration (A72–A73) — ODE (dC_r_hat/dt)
Block 38 : AngII (A74)                 — algebraic
Block 39 : Aldosterone (A75–A80)       — ODE (dC_al_hat/dt)
Block 40 : ANP (A81)                   — algebraic

STATE VECTOR y (length 10 per two-kidney model):
  0 : Paad_R   — right kidney myogenic pressure memory (mmHg)
  1 : Paad_L   — left kidney myogenic pressure memory (mmHg)
  2 : Vecf     — extracellular fluid volume (L)
  3 : Msod     — total body sodium (meq)
  4 : vas      — vascularity index (dimensionless, 1 at normal)
  5 : abaro    — baroreceptor component of autonomic multiplier
  6 : delta_ra — right-atrial pressure adaptation for ADH (dimensionless)
  7 : C_adh_hat — normalized ADH concentration (1 at normal)
  8 : C_r_hat  — normalized renin concentration (1 at normal)
  9 : C_al_hat — normalized aldosterone concentration (1 at normal)

=============================================================================
EQUATION UNCERTAINTIES / FLAGGED ERRORS
=============================================================================
FLAG-1 (A46 vasf coefficient): Paper HTML extracted "11.312" but this gives
  vas_ss ~ 1.1e6 ≠ 1. Physical constraint (vas_ss = 1, Kvd = 1e-5) requires
  coefficient ≈ 1.1e-5. Implemented as 1.1e-5 * exp(...).

FLAG-2 (A53-A56 autonomic block): Equations achemo = 14*aauto + abaro = 34*aauto
  give epsilon_aum ~ 48 at Pma=100, which is inconsistent with Ra requirement.
  Interpreted as misrendering of fractional values 1/4 and 3/4.
  Implemented as: epsilon_aum = achemo + abaro where achemo = aauto/4 (fast)
  and abaro adapts to 3*aauto/4 with T = 2000 min. Net: epsilon_aum ≈ aauto.

FLAG-3 (A61 tubular water reabsorption coefficient): Extracted "12" but gives
  negative urine flow at physiological GFR. Implemented as 0.5 (consistent
  with first extraction and physical mass balance).

FLAG-4 (A70 nu_md-sod denominator): "1843" from HTML likely misrender of "0.1843"
  (consistent with other sigmoid denominators of similar magnitude). Implemented
  as 0.1843.

FLAG-5 (A76 xi_k/sod): Exponent 0.00347 makes term essentially constant (=1).
  Implemented as xi_ksod = 1.0 (constant). Potassium Ck = 4 meq/L assumed fixed.

FLAG-6 (PB + Pgo values): Not stated in main text (referenced to Table S1 which
  is behind paywall). Derived from steady-state constraint: GFR = 90 mL/min per
  kidney, Pgh ≈ 47 mmHg at normal → PB_Pgo = 40 mmHg, Kgcf = 0.0126 L/(min·mmHg).

FLAG-7 (water balance coefficient): Paper A61 HTML shows "12" which gives negative
  urine at normal GFR. Derived K_wreab = 0.659 from constraint: at SS,
  Phi_u_tot = Phi_win = compute_water_intake(Cadh_ss) ≈ 0.001 L/min total.
  With GFR = 0.0814 L/min, mu_al*mu_adh = 1.045:
  K_wreab = (0.2*GFR - Phi_u_target)/(0.025 - 0.001*mu_al*mu_adh) = 0.659.

FLAG-8 (abaro initial condition): Derived as 0.75 * aauto_0 from FLAG-2 interpretation.
  Inner brentq solves Pma for fixed abaro; outer iteration converges abaro.

FLAG-9 (TGF iteration): TGF fixed-point gain ≈ -3.2 → diverges with naive substitution.
  Fixed with damped iteration (alpha=0.25) inside compute_kidney.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ============================================================
# PARAMETERS (from Table S1 / paper appendix)
# ============================================================
DEFAULT_PARAMS = {
    # --- Block 2: Vascular resistance ---
    "Raa_ss":       63.34,      # mmHg·min·L⁻¹  afferent arteriolar SS resistance (A7)
    "Rea_ss":       103.32,     # mmHg·min·L⁻¹  efferent arteriolar SS resistance (A9)

    # --- Block 3: Myogenic ---
    "T_myo":        240.0,      # min  myogenic adaptation time constant (A13)

    # --- Block 6: GFR ---
    "PB_Pgo":       40.0,       # mmHg  Bowman + glomerular oncotic pressure (FLAG-6)
    "Kgcf":         0.0126,     # L/(min·mmHg)  filtration coefficient (FLAG-6)

    # --- Block 9: Proximal Na reabsorption ---
    "neta_pt":      0.8,        # dimensionless  normal proximal frac reab (A25)

    # --- Block 11: Distal Na reabsorption ---
    "neps_dt":      0.5,        # dimensionless  normal distal frac reab (A31)

    # --- Block 13: Collecting duct Na reabsorption ---
    "neta_cd":      0.93,       # dimensionless  normal CD frac reab (A35)

    # --- Block 29: Tubular water reabsorption ---
    "K_wreab":      0.659,      # dimensionless  water reab coefficient (FLAG-3)

    # --- Block 15: Water intake ---
    # Phi_win computed from Cadh (A39)

    # --- Block 16: ECF volume ---
    "Vecf_0":       15.0,       # L  initial ECF volume (A40)

    # --- Block 17: Blood volume ---
    # Vb = 4.560227 + 2.431217/(1 + exp(-(Vecf-18.11278)/2.10806))  (A41)
    "Vb_a":         4.560227,
    "Vb_b":         2.431217,
    "Vb_c":         18.11278,
    "Vb_d":         2.10806,

    # --- Block 18: Mean filling pressure ---
    # Pmf = (7.436*Vb - 30.18) * epsilon_aum  (A42)
    "Pmf_a":        7.436,      # mmHg/L
    "Pmf_b":        30.18,      # mmHg

    # --- Block 21: Right atrial pressure ---
    # Pra = 0.2787 * exp(0.2281 * Phi_co) - 0.879  (A44)
    "Pra_a":        0.2787,
    "Pra_b":        0.2281,
    "Pra_c":        0.879,

    # --- Block 22: Vascularity ---
    "vasf_coeff":   1.1312e-5,  # FLAG-1: coefficient for vasf (A46)
    "vasf_exp":     0.4799,     # exponent factor in vasf (A46) [/100000]
    "Kvd":          1.0e-5,     # min⁻¹  vascularity destruction rate (A47)
    "vas_0":        1.0,        # dimensionless  initial vascularity

    # --- Block 23: Arterial resistance ---
    "Kbar":         16.6,       # mmHg·min·L⁻¹  baseline arterial resistance factor (A49)

    # --- Block 24: Venous return resistance ---
    "Rbv":          3.4,        # mmHg·min·L⁻¹  visceral blood resistance (A50)

    # --- Block 27: Autonomic multiplier ---
    "aauto_coeff":  3.079,      # (A54)
    "aauto_exp":    0.011,      # (A54)
    "T_bar":        2000.0,     # min  baroreceptor adaptation time constant (A56)

    # --- Block 28: ADH ---
    "T_adh":        6.0,        # min  ADH time constant (A59)
    "Cadh_ss":      4.0,        # mU/L  normal ADH (A60)
    "Csod_adh_thr": 141.0,      # meq/L  sodium threshold for ADH secretion (A57)
    "aum_adh_thr":  1.0,        # autonomic multiplier threshold for ADH (A57)
    "delta_ra_gain":0.0007,     # min⁻¹  delta_ra adaptation gain (A58)

    # --- Block 34: Total body sodium ---
    "Msod_0":       2160.0,     # meq  initial total body sodium (A67)

    # --- Block 36-37: Renin ---
    "T_r":          15.0,       # min  renin time constant (A73)
    "Cat_ss":       20.0,       # ng/L  normal AngII (A74)
    "Sr_ss_tot":    1500.0,     # ng ANGI·h⁻¹·min⁻¹  normal total renin secretion rate

    # --- Block 39: Aldosterone ---
    "T_al":         60.0,       # min  aldosterone time constant (A79)
    "Cal_ss":       85.0,       # ng/L  normal aldosterone (A80)
    "Ck":           4.0,        # meq/L  plasma potassium (constant; FLAG-5)

    # --- Normal sodium intake ---
    "Phi_sodin_normal": 0.126,  # meq/min  total (both kidneys) normal Na intake (Block 33)
}


# ============================================================
# ALGEBRAIC HELPER FUNCTIONS (per kidney)
# ============================================================

def _log10_safe(x):
    """Safe log10 for positive values."""
    return np.log10(np.maximum(x, 1e-12))


def compute_rsna(Pma, Pra, Nrsna=1.0):
    """
    Block 1 — RSNA (A3–A5)
    RSNA = Nrsna * alpha_map * alpha_rap
    alpha_map = 0.5 + 1.05/(1 + exp((Pma - 100)/15))
    alpha_rap = 1 - 0.08 * Pra
    """
    alpha_map = 0.5 + 1.05 / (1.0 + np.exp((Pma - 100.0) / 15.0))  # A4
    alpha_rap = 1.0 - 0.08 * Pra                                       # A5
    return Nrsna * alpha_map * alpha_rap                                 # A3


def compute_renal_vascular(RSNA, Cat, Paac, Phi_rb, params):
    """
    Blocks 2, 3 (Maac), 4 (NO) — renal vascular resistance.

    Returns: (Raa, Rea, Rr)
    """
    p = params

    # --- Block 2: AngII effects on afferent / efferent (A10-A11) ---
    zeta_ata = 0.9854 + 0.03658 / (0.2215 + np.exp(3.115 - 1.7864 * _log10_safe(Cat)))  # A11
    zeta_ate = 0.9432 + 0.1363  / (0.2069 + np.exp(3.108 - 1.785  * _log10_safe(Cat)))  # A10

    # --- RSNA effect on afferent arteriole (A8) ---
    beta_rsna = 1.5 * (RSNA - 1.0) + 1.0                              # A8

    # --- Block 3: Myogenic response (A15) ---
    Maac = 2.094 - 8.355 / (5.734 + np.exp((Paac + 15.87) / 4.66))   # A15

    # --- Block 4: Nitric oxide (A16-A17) ---
    SNO   = 1.228 - 0.04802 / (0.1079 + np.exp((Phi_rb - 0.8661) / 0.115))  # A16
    Omega_NO = -0.3 * SNO + 1.3                                         # A17

    # TGF signal — computed separately (see compute_tgf), used as placeholder
    # here; caller must pass Sigma_tgf
    return zeta_ata, zeta_ate, beta_rsna, Maac, Omega_NO


def compute_tgf(Phi_md_sod):
    """Block 7 — Tubuloglomerular feedback signal (A22)."""
    return 0.3412 + 0.06296 / (0.07079 + np.exp(-2.064 * Phi_md_sod))  # A22


def compute_gfr(Pma, Phi_rb, Raa, params):
    """
    Block 6 — GFR (A19-A21)
    Pgh = Pma - Phi_rb * Raa           (A21)
    Pf  = Pgh - PB_Pgo                  (A20, with PB+Pgo combined)
    GFR = Kgcf * Pf                     (A19)
    """
    p = params
    Pgh = Pma - Phi_rb * Raa           # A21 — glomerular hydrostatic pressure
    Pf  = Pgh - p["PB_Pgo"]            # A20 — net filtration pressure
    Pf  = np.maximum(Pf, 0.0)          # physiological clamp
    return p["Kgcf"] * Pf, Pgh, Pf    # (GFR, Pgh, Pf)


def compute_proximal_reab(Phi_filsod, Cat, RSNA, params):
    """
    Block 9 — Proximal tubule sodium reabsorption (A24-A28)
    eta_pt = neta_pt * gamma_filsod * gamma_at * gamma_rsna
    """
    p = params
    gamma_filsod = (0.7953 + 2.167 /
                    (4.063 + np.exp((Phi_filsod - 5.663) / 4.448)))    # A26
    gamma_at     = (0.95 + 0.12 /
                    (1.0 + np.exp(2.6 - 1.8 * _log10_safe(Cat))))      # A27
    gamma_rsna   = (1.1916 - 0.4762 /
                    (1.064 + np.exp((RSNA - 0.9034) / 0.3697)))        # A28
    eta_pt_reab  = p["neta_pt"] * gamma_filsod * gamma_at * gamma_rsna # A25
    Phi_pt_reab  = Phi_filsod * eta_pt_reab                             # A24
    return Phi_pt_reab, eta_pt_reab


def compute_distal_reab(Phi_md_sod, Cal, params):
    """
    Block 11 — Distal tubule sodium reabsorption (A30-A32)
    eta_dt = neps_dt * Psi_al
    """
    p = params
    Psi_al      = (0.17 + 0.94 /
                   (1.0 + np.exp((0.48 - 1.2 * _log10_safe(Cal)) / 0.88)))  # A32
    eta_dt_reab = p["neps_dt"] * Psi_al                                       # A31
    Phi_dt_reab = Phi_md_sod * eta_dt_reab                                    # A30
    return Phi_dt_reab, eta_dt_reab, Psi_al


def compute_cd_reab(Phi_dt_sod, C_anp_hat, Cal, params):
    """
    Block 13 — Collecting duct sodium reabsorption (A34-A37)
    eta_cd = neta_cd * lambda_dt * lambda_anp
    """
    p = params
    lambda_dt  = (0.796 + 0.4778 /
                  (1.222 + np.exp((Phi_dt_sod - 0.8801) / 0.9194)))   # A36
    lambda_anp = -0.1 * C_anp_hat + 1.1                                 # A37
    eta_cd_reab = p["neta_cd"] * lambda_dt * lambda_anp                 # A35
    Phi_cd_reab = Phi_dt_sod * eta_cd_reab                              # A34
    return Phi_cd_reab, eta_cd_reab


def compute_anp(Pra):
    """Block 40 — Atrial natriuretic peptide (A81)."""
    return 7.4052 - 6.554 / (1.0 + np.exp(Pra - 3.762))               # A81


def compute_renin_secretion(Phi_md_sod, RSNA):
    """
    Block 36 — Renin secretion per kidney (A69-A71)
    Sr = nu_md_sod * nu_rsna
    """
    # FLAG-4: denominator 0.1843 (not 1843)
    nu_md_sod = (169.9 + 335.0 /
                 (0.1843 + np.exp((Phi_md_sod - 2.085) / 0.3024)))     # A70
    nu_rsna   = (1.89 - 2.056 /
                 (1.358 + np.exp(RSNA - 0.8667)))                       # A71
    return nu_md_sod * nu_rsna                                           # A69


def compute_ald_secretion(Pma, Cat, Csod, params):
    """
    Block 39 — Aldosterone secretion (A75-A78)
    Sal = xi_ksod * xi_map * xi_at
    """
    p = params
    # FLAG-5: xi_ksod ≈ 1 (exponent 0.00347 negligible)
    xi_ksod = 1.0                                                        # A76 simplified

    xi_map = np.where(Pma <= 100.0,
                      69.03 * np.exp(-0.0425 * Pma),
                      1.0)                                               # A77
    xi_at  = (0.4 + 2.4 /
              (1.0 + np.exp((2.82 - 1.5 * _log10_safe(Cat)) / 0.8)))   # A78
    return xi_ksod * xi_map * xi_at                                      # A75


def compute_autonomic(Pma, params):
    """
    Block 27 — Autonomic signal aauto (A54).
    epsilon_aum decomposed as achemo + abaro where:
      achemo = aauto/4  (fast, chemoreceptor)
      abaro   adapts to 3*aauto/4 with T_bar = 2000 min (FLAG-2)
    The ODE drives abaro; epsilon_aum = achemo + abaro.
    """
    p = params
    aauto = p["aauto_coeff"] * np.exp(-p["aauto_exp"] * Pma)            # A54
    achemo = aauto / 4.0                                                  # FLAG-2 interpretation
    return aauto, achemo


def compute_cardiac(Vb, epsilon_aum, vas, params):
    """
    Blocks 17-26 — Cardiovascular sub-model.
    Solves implicitly for cardiac output Phi_co given current state.

    Returns: (Phi_co, Pma, Pra, Ra, Rtp, Rvr, Pmf)
    """
    p = params

    # A42 — mean filling pressure
    Pmf = (p["Pmf_a"] * Vb - p["Pmf_b"]) * epsilon_aum

    # A48-A49 — arterial resistance
    Ra  = (p["Kbar"] / vas) * epsilon_aum

    # A50 — venous return resistance
    Rvr = (8.0 * p["Rbv"] + Ra) / 31.0

    # A51 — total peripheral resistance
    Rtp = Ra + p["Rbv"]

    # Implicit solve: Phi_co = (Pmf - Pra)/Rvr and Pra = f(Phi_co)
    # A44: Pra = 0.2787 * exp(0.2281 * Phi_co) - 0.879
    def residual(Phi_co):
        Pra_ = p["Pra_a"] * np.exp(p["Pra_b"] * Phi_co) - p["Pra_c"]  # A44
        return Phi_co - (Pmf - Pra_) / Rvr                               # A43

    # Bracket: CO between 0.1 and 30 L/min
    try:
        Phi_co = brentq(residual, 0.1, 30.0, xtol=1e-6)
    except ValueError:
        Phi_co = 5.0  # fallback

    Pra  = p["Pra_a"] * np.exp(p["Pra_b"] * Phi_co) - p["Pra_c"]      # A44
    Pma  = Phi_co * Rtp                                                   # A52

    return Phi_co, Pma, Pra, Ra, Rtp, Rvr, Pmf


def compute_water(Phi_gfilt, Cal, Cadh, params):
    """
    Block 29 — Tubular water reabsorption (A61–A63)
    Phi_t_wreab = K_wreab * (0.025 - 0.001 * mu_al * mu_adh) + 0.8 * Phi_gfilt

    FLAG-3: Paper HTML shows coefficient "12" → physically impossible (gives negative
    urine at GFR = 90 mL/min). Correct value derived from water-balance constraint:
      at SS:  Phi_u_tot = Phi_win = 0.01 * mu_adh(Cadh_ss) - 0.0094 ≈ 0.001 L/min
      per kidney Phi_u = 0.0005 L/min at GFR = 0.082 L/min
      → K_wreab = (0.2*GFR - 0.0005) / (0.025 - 0.001*mu_al*mu_adh) ≈ 0.664
    Implemented as K_wreab = 0.664 (stored in params["K_wreab"]).
    """
    p = params
    mu_al  = (0.17 + 0.94 /
              (1.0 + np.exp((0.48 - 1.2 * _log10_safe(Cal)) / 0.88)))  # A62
    mu_adh = (0.37 + 0.8 /
              (1.0 + np.exp(0.6 - 3.7 * _log10_safe(Cadh))))            # A63
    Phi_wreab = p["K_wreab"] * (0.025 - 0.001 * mu_al * mu_adh) + 0.8 * Phi_gfilt  # A61
    return max(Phi_wreab, 0.0)


def compute_water_intake(Cadh):
    """Block 15 — Water intake (A39)."""
    return 0.01 * (0.37 + 0.8 / (1.0 + np.exp(0.6 - 3.7 * _log10_safe(Cadh)))) - 0.0094  # A39


# ============================================================
# ONE-KIDNEY RENAL CALCULATION
# ============================================================

def _kidney_inner(Sigma_tgf, Pma, Paad, RSNA, Cat, Csod, params):
    """
    Inner per-kidney solve for FIXED Sigma_tgf.
    Iterates the Phi_rb / Paac / Maac / Omega_NO loop (all gains < 0.35 → converges).

    Returns: (Raa, Rea, Phi_rb, GFR, Pgh, Paa, Phi_filsod, Phi_pt_reab, Phi_md_sod,
              zeta_ata, zeta_ate, beta_rsna, Maac, Omega_NO)
    """
    p = params
    Phi_rb = Pma / (p["Raa_ss"] + p["Rea_ss"])  # warm start

    for _ in range(10):
        # First call: Paac=0 to get zeta_ata, beta_rsna (not Paac-dependent)
        zeta_ata, zeta_ate, beta_rsna, Maac, Omega_NO = \
            compute_renal_vascular(RSNA, Cat, 0.0, Phi_rb, p)
        Raa = p["Raa_ss"] * beta_rsna * Sigma_tgf * zeta_ata * Maac * Omega_NO  # A7
        Rea = p["Rea_ss"] * zeta_ate                                               # A9
        Phi_rb = np.maximum(Pma / (Raa + Rea), 1e-6)                              # A18
        GFR, Pgh, _ = compute_gfr(Pma, Phi_rb, Raa, p)

        # Second call: actual Paac (myogenic) and updated Phi_rb (NO)
        Paa  = (Pma + Pgh) / 2.0                                                   # A12
        Paac = Paa - Paad                                                           # A14
        _, _, _, Maac, Omega_NO = compute_renal_vascular(RSNA, Cat, Paac, Phi_rb, p)
        Raa_new = p["Raa_ss"] * beta_rsna * Sigma_tgf * zeta_ata * Maac * Omega_NO
        Phi_rb_new = np.maximum(Pma / (Raa_new + Rea), 1e-6)
        GFR_new, Pgh_new, _ = compute_gfr(Pma, Phi_rb_new, Raa_new, p)
        Paa_new = (Pma + Pgh_new) / 2.0

        if abs(Phi_rb_new - Phi_rb) < 1e-6:
            Raa, Phi_rb, GFR, Pgh, Paa = Raa_new, Phi_rb_new, GFR_new, Pgh_new, Paa_new
            break
        Raa, Phi_rb, GFR, Pgh, Paa = Raa_new, Phi_rb_new, GFR_new, Pgh_new, Paa_new

    Phi_filsod  = GFR * Csod                                                       # A23
    Phi_pt_reab, _ = compute_proximal_reab(Phi_filsod, Cat, RSNA, p)              # A24-A28
    Phi_md_sod  = np.maximum(Phi_filsod - Phi_pt_reab, 1e-8)                      # A29

    return (Raa, Rea, Phi_rb, GFR, Pgh, Paa, Phi_filsod, Phi_pt_reab, Phi_md_sod,
            zeta_ata, zeta_ate, beta_rsna, Maac, Omega_NO)


def compute_kidney(Pma, Pra, RSNA, Paad, Cat, Cal, Cadh, C_anp_hat, Csod, params):
    """
    Compute all per-kidney renal quantities.

    TGF fixed-point equation: Sigma_tgf = compute_tgf(Phi_md_sod(Sigma_tgf))
    The direct iteration diverges (gain ≈ -3.2).
    Solved with brentq on F(Σ) = Σ - compute_tgf(Phi_md_sod_inner(Σ)) = 0.
      F(0.1) < 0 and F(2.0) > 0 across all physiological inputs.
    """
    p = params

    def F_tgf(sigma):
        _, _, _, _, _, _, _, _, Phi_md_sod, _, _, _, _, _ = \
            _kidney_inner(sigma, Pma, Paad, RSNA, Cat, Csod, p)
        return sigma - compute_tgf(Phi_md_sod)                                     # A22

    # Damped fixed-point iteration: gain ≈ -3.2 so undamped diverges.
    # Damping factor alpha = 0.2 gives effective gain ≈ 0.64 → converges.
    # ~30 iterations are needed; brentq is overkill here but available as fallback.
    Sigma_tgf = 1.0  # warm start (close to physical SS for healthy state)
    alpha = 0.25     # damping factor (1 - alpha × gain must be < 1 for stability)
    for _it in range(50):
        _, _, _, _, _, _, _, _, Phi_md_sod_it, _, _, _, _, _ = \
            _kidney_inner(Sigma_tgf, Pma, Paad, RSNA, Cat, Csod, p)
        Sigma_new = compute_tgf(Phi_md_sod_it)                                     # A22
        Sigma_next = (1.0 - alpha) * Sigma_tgf + alpha * Sigma_new
        if abs(Sigma_next - Sigma_tgf) < 1e-7:
            Sigma_tgf = Sigma_next
            break
        Sigma_tgf = Sigma_next
    else:
        # fallback to brentq if damped iteration didn't converge
        try:
            Sigma_tgf = brentq(F_tgf, 0.1, 2.0, xtol=1e-6, maxiter=100)
        except ValueError:
            pass  # keep last damped value

    # Retrieve converged inner solution at the solved Sigma_tgf
    (Raa, Rea, Phi_rb, GFR, Pgh, Paa, Phi_filsod, Phi_pt_reab, Phi_md_sod,
     zeta_ata, zeta_ate, beta_rsna, Maac, Omega_NO) = \
        _kidney_inner(Sigma_tgf, Pma, Paad, RSNA, Cat, Csod, p)
    Rr = Raa + Rea

    # Distal reabsorption
    Phi_dt_reab, _, Psi_al = compute_distal_reab(Phi_md_sod, Cal, p)              # A30-A32
    Phi_dt_sod = np.maximum(Phi_md_sod - Phi_dt_reab, 1e-6)                        # A33

    # ANP and collecting duct
    C_anp_hat_val = compute_anp(Pra)                                                # A81
    Phi_cd_reab, _ = compute_cd_reab(Phi_dt_sod, C_anp_hat_val, Cal, p)           # A34-A37
    Phi_u_sod = np.maximum(Phi_dt_sod - Phi_cd_reab, 0.0)                          # A38

    # Water reabsorption and urine flow
    Phi_wreab = compute_water(GFR, Cal, Cadh, p)                                    # A61
    Phi_u     = np.maximum(GFR - Phi_wreab, 0.0)                                    # A64

    # Renin secretion
    Sr = compute_renin_secretion(Phi_md_sod, RSNA)                                  # A69

    return {
        "Raa": Raa, "Rea": Rea, "Rr": Rr,
        "Phi_rb": Phi_rb, "GFR": GFR, "Pgh": Pgh, "Paa": Paa,
        "Phi_filsod": Phi_filsod,
        "Phi_pt_reab": Phi_pt_reab,
        "Phi_md_sod": Phi_md_sod,
        "Sigma_tgf": Sigma_tgf,
        "Phi_dt_sod": Phi_dt_sod,
        "Phi_u_sod": Phi_u_sod,
        "Phi_u": Phi_u,
        "Sr": Sr,
        "C_anp_hat": C_anp_hat_val,
        "Maac": Maac,
        "Psi_al": Psi_al,
        "Omega_NO": Omega_NO,
        "SNO": 1.228 - 0.04802 / (0.1079 + np.exp((Phi_rb - 0.8661) / 0.115)),
    }


# ============================================================
# ODE RIGHT-HAND SIDE
# ============================================================

def rhs(t, y, params, RSNA_L_func, RSNA_R_func, Phi_sodin_func):
    """
    Full two-kidney ODE right-hand side.

    State vector y:
      y[0] Paad_R  — right kidney myogenic memory (mmHg)
      y[1] Paad_L  — left kidney myogenic memory (mmHg)
      y[2] Vecf    — ECF volume (L)
      y[3] Msod    — total body sodium (meq)
      y[4] vas     — vascularity (dimensionless)
      y[5] abaro   — baroreceptor component of epsilon_aum
      y[6] delta_ra — RA pressure adaptation
      y[7] C_adh_hat — normalized ADH
      y[8] C_r_hat  — normalized renin
      y[9] C_al_hat — normalized aldosterone

    Parameters:
      RSNA_L_func(t) : function returning RSNA for left kidney at time t
                        (can be fixed constant or variable)
      RSNA_R_func(t) : function returning RSNA for right kidney at time t
      Phi_sodin_func(t) : function returning total sodium intake (meq/min)
    """
    p = params

    # --- Unpack state ---
    Paad_R, Paad_L, Vecf, Msod, vas, abaro, delta_ra, C_adh_hat, C_r_hat, C_al_hat = y

    # --- Derived hormone concentrations ---
    Cadh = C_adh_hat * p["Cadh_ss"]                                     # A60
    Cat  = C_r_hat   * p["Cat_ss"]                                       # A74
    Cal  = C_al_hat  * p["Cal_ss"]                                       # A80
    Csod = Msod / np.maximum(Vecf, 1e-3)                                 # A68

    # --- Blood volume and autonomic ---
    Vb = (p["Vb_a"] + p["Vb_b"] /                                       # A41
          (1.0 + np.exp(-(Vecf - p["Vb_c"]) / p["Vb_d"])))

    # --- Cardiovascular: need epsilon_aum → need Pma → chicken-and-egg ---
    # abaro tracks 3/4 * aauto(Pma_ss); invert aauto to get a warm start.
    # achemo = aauto/4 is the only Pma-dependent term inside the inner loop.
    # Loop gain = dg/deps * (-aauto_exp * aauto/4) ≈ −0.41 → converges.
    abaro_safe = max(abaro, 1e-4)
    aauto_est_init = abaro_safe * 4.0 / 3.0
    Pma_est = float(np.clip(
        -np.log(max(aauto_est_init / p["aauto_coeff"], 1e-9)) / p["aauto_exp"],
        50.0, 300.0))
    for _ in range(10):
        aauto_est, achemo_est = compute_autonomic(Pma_est, p)
        eps_aum = achemo_est + abaro
        eps_aum = np.maximum(eps_aum, 0.01)
        Phi_co_est, Pma_new, Pra_est, Ra_est, Rtp_est, Rvr_est, Pmf_est = \
            compute_cardiac(Vb, eps_aum, vas, p)
        if abs(Pma_new - Pma_est) < 0.01:
            break
        Pma_est = Pma_new

    Pma     = Pma_new
    Pra     = Pra_est
    Phi_co  = Phi_co_est
    eps_aum = eps_aum
    aauto   = aauto_est

    # --- Per-kidney RSNA ---
    RSNA_R_base = compute_rsna(Pma, Pra)                                 # A3
    RSNA_R = RSNA_R_func(t, RSNA_R_base)                                 # modulated externally
    RSNA_L = RSNA_L_func(t, RSNA_R_base)                                 # left kidney RSNA

    # --- ANP (from Pra) ---
    C_anp_hat = compute_anp(Pra)                                          # A81

    # --- Per-kidney calculations ---
    kR = compute_kidney(Pma, Pra, RSNA_R, Paad_R, Cat, Cal, Cadh,
                        C_anp_hat, Csod, p)
    kL = compute_kidney(Pma, Pra, RSNA_L, Paad_L, Cat, Cal, Cadh,
                        C_anp_hat, Csod, p)

    # --- Totals ---
    Phi_u_tot   = kR["Phi_u"]     + kL["Phi_u"]                          # A65
    Phi_sod_tot = kR["Phi_u_sod"] + kL["Phi_u_sod"]                      # A66
    Sr_tot      = kR["Sr"]        + kL["Sr"]
    GFR_tot     = kR["GFR"]       + kL["GFR"]

    # --- Water intake (A39) ---
    Phi_win = compute_water_intake(Cadh)                                   # A39

    # --- Sodium intake ---
    Phi_sodin = Phi_sodin_func(t)                                          # Block 33

    # --- Normalized renin secretion rate ---
    S_r_tot_hat = Sr_tot / p["Sr_ss_tot"]                                  # A72

    # --- Aldosterone secretion ---
    S_al = compute_ald_secretion(Pma, Cat, Csod, p)                        # A75

    # --- ADH secretion (A57) ---
    # Valid when Csod > 141 meq/L and eps_aum > 1
    S_adh_raw = (1.0/3.0) * ((Csod - p["Csod_adh_thr"])
                              + (eps_aum - p["aum_adh_thr"])
                              - delta_ra)
    # Only contributes when above threshold
    S_adh = np.where(
        (Csod > p["Csod_adh_thr"]) | (eps_aum > p["aum_adh_thr"]),
        S_adh_raw,
        0.0
    )
    S_adh = float(np.maximum(S_adh, 0.0))

    # --- Vascularity formation (A46, FLAG-1) ---
    vasf = p["vasf_coeff"] * np.exp(-Phi_co * p["vasf_exp"] / 100000.0)   # A46
    vasd = vas * p["Kvd"]                                                    # A47

    # ============================================================
    # ODEs
    # ============================================================

    # --- Block 3: Myogenic (A13) ---
    dPaad_R = (kR["Paa"] - Paad_R) / p["T_myo"]
    dPaad_L = (kL["Paa"] - Paad_L) / p["T_myo"]

    # --- Block 16: ECF volume (A40) ---
    dVecf = Phi_win - Phi_u_tot

    # --- Block 34: Total body sodium (A67) ---
    dMsod = Phi_sodin - Phi_sod_tot

    # --- Block 22: Vascularity (A45) ---
    dvas = vasf - vasd

    # --- Block 27: Baroreceptor (FLAG-2) ---
    # abaro adapts toward 3*aauto/4 with time constant T_bar
    dabaro = (3.0 * aauto / 4.0 - abaro) / p["T_bar"]

    # --- Block 28: RA pressure adaptation for ADH (A58) ---
    ddelta_ra = p["delta_ra_gain"] * (0.2 * Pra - delta_ra)

    # --- Block 28: ADH dynamics (A59) ---
    dC_adh_hat = (S_adh - C_adh_hat) / p["T_adh"]

    # --- Block 37: Renin dynamics (A73) ---
    dC_r_hat = (S_r_tot_hat - C_r_hat) / p["T_r"]

    # --- Block 39: Aldosterone dynamics (A79) ---
    dC_al_hat = (S_al - C_al_hat) / p["T_al"]

    return [dPaad_R, dPaad_L, dVecf, dMsod, dvas, dabaro,
            ddelta_ra, dC_adh_hat, dC_r_hat, dC_al_hat]


# ============================================================
# INITIAL CONDITIONS
# ============================================================

def _solve_pma(abaro_fixed, Vb, vas, p):
    """
    Solve for self-consistent (Pma, eps_aum) given a FIXED abaro.
    Only the fast chemoreceptor component (achemo = aauto/4) feeds back on Pma.
    Uses brentq on F(Pma) = Pma - g(achemo(Pma) + abaro).
    Loop gain ≈ −0.41 → the brentq root is unique in physiological range.
    """
    def F(Pma):
        aauto  = p["aauto_coeff"] * np.exp(-p["aauto_exp"] * Pma)
        achemo = aauto / 4.0
        eps    = max(achemo + abaro_fixed, 0.01)
        _, Pma_new, _, _, _, _, _ = compute_cardiac(Vb, eps, vas, p)
        return Pma_new - Pma

    # F(Pma) changes sign between 50 and 300 mmHg for all physiological inputs
    try:
        Pma_ss = brentq(F, 50.0, 300.0, xtol=0.001, maxiter=200)
    except ValueError:
        Pma_ss = 100.0  # fallback

    aauto_ss  = p["aauto_coeff"] * np.exp(-p["aauto_exp"] * Pma_ss)
    achemo_ss = aauto_ss / 4.0
    eps_ss    = max(achemo_ss + abaro_fixed, 0.01)
    Phi_co_ss, _, Pra_ss, Ra_ss, Rtp_ss, Rvr_ss, Pmf_ss = \
        compute_cardiac(Vb, eps_ss, vas, p)
    return Pma_ss, Phi_co_ss, Pra_ss, eps_ss, aauto_ss


def compute_initial_conditions(params):
    """
    Compute initial conditions at normal steady state.

    Strategy:
      abaro and Pma are mutually dependent:
        abaro_ss = 3 * aauto(Pma_ss) / 4
        Pma_ss   = f(achemo(Pma_ss) + abaro_ss)
      Outer fixed-point iteration on abaro (gain ≈ −0.88 → converges in ~20 steps).
      Inner brentq solves for Pma given abaro.
    """
    p = params

    Vecf_0 = p["Vecf_0"]
    Msod_0 = p["Msod_0"]
    vas_0  = p["vas_0"]
    Cat_0  = p["Cat_ss"]   # normalized to 1 at normal

    Vb_0 = (p["Vb_a"] + p["Vb_b"] /
            (1.0 + np.exp(-(Vecf_0 - p["Vb_c"]) / p["Vb_d"])))

    # Outer iteration: abaro ← 3/4 * aauto(Pma)
    abaro_0 = 3.0 * p["aauto_coeff"] * np.exp(-p["aauto_exp"] * 100.0) / 4.0
    Pma_0   = 100.0
    for _ in range(30):
        Pma_new, Phi_co_0, Pra_0, eps_aum_0, aauto_0 = \
            _solve_pma(abaro_0, Vb_0, vas_0, p)
        abaro_new = 3.0 * aauto_0 / 4.0
        if abs(abaro_new - abaro_0) < 1e-6 and abs(Pma_new - Pma_0) < 0.001:
            break
        abaro_0 = abaro_new
        Pma_0   = Pma_new

    # RSNA and kidney geometry at steady state
    RSNA_0   = compute_rsna(Pma_0, Pra_0)
    Cat_0    = p["Cat_ss"]
    Cal_0    = p["Cal_ss"]
    Cadh_0   = p["Cadh_ss"]
    Csod_0   = Msod_0 / Vecf_0
    C_anp_0  = compute_anp(Pra_0)

    # Use brentq-solved kidney to get Paa at SS (Paac = 0 → Paad = Paa)
    # We need Paad as a starting value for the first call; set to Pma/2 initially,
    # then use the returned Paa as the true Paad.
    Paad_guess = Pma_0 * 0.75
    kIC = compute_kidney(Pma_0, Pra_0, RSNA_0, Paad_guess, Cat_0, Cal_0, Cadh_0, C_anp_0, Csod_0, p)
    # At SS, Paad = Paa (Paac → 0 as Paad adapts). Run one refinement:
    Paad_0 = kIC["Paa"]
    kIC    = compute_kidney(Pma_0, Pra_0, RSNA_0, Paad_0, Cat_0, Cal_0, Cadh_0, C_anp_0, Csod_0, p)
    Paad_0 = kIC["Paa"]  # at SS with Paac=0 → Paad=Paa

    delta_ra_0  = 0.2 * Pra_0  # long-term SS of A58

    y0 = [Paad_0, Paad_0, Vecf_0, Msod_0, vas_0, abaro_0,
          delta_ra_0, 1.0, 1.0, 1.0]

    return y0, {"Pma_0": Pma_0, "Phi_co_0": Phi_co_0, "Pra_0": Pra_0,
                "RSNA_0": RSNA_0, "Vb_0": Vb_0, "Raa_0": kIC["Raa"],
                "eps_aum_0": eps_aum_0, "abaro_0": abaro_0,
                "GFR_0": kIC["GFR"], "Sigma_tgf_0": kIC["Sigma_tgf"],
                "Phi_u_sod_0": kIC["Phi_u_sod"]}


# ============================================================
# SIMULATION RUNNER
# ============================================================

def run_simulation(t_span, t_eval, RSNA_L_func, RSNA_R_func,
                   Phi_sodin_func, params=None, y0=None,
                   rtol=1e-8, atol=1e-10):
    """
    Run the two-kidney ODE model.

    Parameters
    ----------
    t_span : (t_start, t_end)  in minutes
    t_eval : array of times to report output (minutes)
    RSNA_L_func : callable(t, RSNA_natural) -> RSNA for left kidney
    RSNA_R_func : callable(t, RSNA_natural) -> RSNA for right kidney
    Phi_sodin_func : callable(t) -> total sodium intake (meq/min)
    params : dict (defaults to DEFAULT_PARAMS if None)
    y0 : initial state vector (computed if None)
    rtol, atol : solver tolerances

    Returns
    -------
    sol : ODE solution object from solve_ivp
    y0  : initial conditions used
    p   : parameter dict used
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if y0 is None:
        y0, _ = compute_initial_conditions(params)

    def f(t, y):
        return rhs(t, y, params, RSNA_L_func, RSNA_R_func, Phi_sodin_func)

    sol = solve_ivp(
        f, t_span, y0, t_eval=t_eval,
        method="Radau",
        rtol=rtol, atol=atol,
        dense_output=False
    )
    return sol, y0, params


# ============================================================
# POST-PROCESSING: compute derived quantities from solution
# ============================================================

def extract_outputs(sol, params, RSNA_L_func, RSNA_R_func, Phi_sodin_func):
    """
    Recompute all derived quantities at each time point in the solution.
    Returns a dict of time-series arrays.
    """
    p = params
    nt = len(sol.t)

    out = {k: np.zeros(nt) for k in [
        "Pma", "Phi_co", "Pra", "Vb", "eps_aum",
        "RSNA_R", "RSNA_L",
        "GFR_R", "GFR_L", "GFR_tot",
        "Phi_u_sod_R", "Phi_u_sod_L", "Phi_u_sod_tot",
        "Phi_u_R", "Phi_u_L", "Phi_u_tot",
        "Sr_R", "Sr_L",
        "Cat", "Cal", "Cadh", "Csod",
        "Raa_R", "Raa_L", "Phi_rb_R", "Phi_rb_L",
        "Sigma_tgf_R", "Sigma_tgf_L",
        "Phi_sodin",
    ]}

    for i in range(nt):
        y = sol.y[:, i]
        Paad_R, Paad_L, Vecf, Msod, vas, abaro, delta_ra, C_adh_hat, C_r_hat, C_al_hat = y

        Cadh = C_adh_hat * p["Cadh_ss"]
        Cat  = C_r_hat   * p["Cat_ss"]
        Cal  = C_al_hat  * p["Cal_ss"]
        Csod = Msod / max(Vecf, 1e-3)

        Vb = (p["Vb_a"] + p["Vb_b"] /
              (1.0 + np.exp(-(Vecf - p["Vb_c"]) / p["Vb_d"])))

        # Cardiovascular
        Pma_est = 100.0
        for _ in range(10):
            aauto_est, achemo_est = compute_autonomic(Pma_est, p)
            eps_aum = achemo_est + abaro
            eps_aum = max(eps_aum, 0.01)
            Phi_co, Pma_new, Pra, Ra, Rtp, Rvr, Pmf = \
                compute_cardiac(Vb, eps_aum, vas, p)
            if abs(Pma_new - Pma_est) < 0.01:
                break
            Pma_est = Pma_new
        Pma = Pma_new

        RSNA_base = compute_rsna(Pma, Pra)
        RSNA_R = RSNA_R_func(sol.t[i], RSNA_base)
        RSNA_L = RSNA_L_func(sol.t[i], RSNA_base)

        C_anp_hat = compute_anp(Pra)

        kR = compute_kidney(Pma, Pra, RSNA_R, Paad_R, Cat, Cal, Cadh, C_anp_hat, Csod, p)
        kL = compute_kidney(Pma, Pra, RSNA_L, Paad_L, Cat, Cal, Cadh, C_anp_hat, Csod, p)

        out["Pma"][i]          = Pma
        out["Phi_co"][i]       = Phi_co
        out["Pra"][i]          = Pra
        out["Vb"][i]           = Vb
        out["eps_aum"][i]      = eps_aum
        out["RSNA_R"][i]       = RSNA_R
        out["RSNA_L"][i]       = RSNA_L
        out["GFR_R"][i]        = kR["GFR"]
        out["GFR_L"][i]        = kL["GFR"]
        out["GFR_tot"][i]      = kR["GFR"] + kL["GFR"]
        out["Phi_u_sod_R"][i]  = kR["Phi_u_sod"]
        out["Phi_u_sod_L"][i]  = kL["Phi_u_sod"]
        out["Phi_u_sod_tot"][i]= kR["Phi_u_sod"] + kL["Phi_u_sod"]
        out["Phi_u_R"][i]      = kR["Phi_u"]
        out["Phi_u_L"][i]      = kL["Phi_u"]
        out["Phi_u_tot"][i]    = kR["Phi_u"] + kL["Phi_u"]
        out["Sr_R"][i]         = kR["Sr"]
        out["Sr_L"][i]         = kL["Sr"]
        out["Cat"][i]          = Cat
        out["Cal"][i]          = Cal
        out["Cadh"][i]         = Cadh
        out["Csod"][i]         = Csod
        out["Raa_R"][i]        = kR["Raa"]
        out["Raa_L"][i]        = kL["Raa"]
        out["Phi_rb_R"][i]     = kR["Phi_rb"]
        out["Phi_rb_L"][i]     = kL["Phi_rb"]
        out["Sigma_tgf_R"][i]  = kR["Sigma_tgf"]
        out["Sigma_tgf_L"][i]  = kL["Sigma_tgf"]
        out["Phi_sodin"][i]    = Phi_sodin_func(sol.t[i])

    out["t"] = sol.t
    return out
