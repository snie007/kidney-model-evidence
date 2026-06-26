"""
Gebremichael et al. 2018 — Toxicological Sciences 162(1):200-211
DOI: 10.1093/toxsci/kfx239  |  PMID: 29126144

"Multiscale Mathematical Model of Drug-Induced Proximal Tubule Injury:
 Linking Urinary Biomarkers to Epithelial Cell Injury and Renal Dysfunction"

Python port (scipy) from the R/rxode2 reference implementation.
Reference output (R): 2.5 mg/kg rat, 250g:
  peak C_plasma = 1257.54 ng/mL at t=60 min
  peak Inj      = 0.411  at day 2.2
  peak Nec      = 0.412  at day 4.0
  peak Kim-1    = 50.1x  at day 6.0
  peak aGST     = 22.4x  at day 4.0
  peak sCr      = 1.17x  at day 3.3

Units: time=minutes, amounts=ng, concentrations=ng/mL
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# PARAMETERS — all from supplement Tables S1/S2/S3
# ---------------------------------------------------------------------------

# Table S1: cisplatin 2-compartment PK (250g rat, IP dosing)
PK = dict(
    ka_drug  = 2.2e-4,   # min⁻¹    peritoneal absorption
    k12_drug = 0.001,    # min⁻¹    central→peripheral
    k21_drug = 0.001,    # min⁻¹    peripheral→central
    ke_drug  = 0.107,    # min⁻¹    elimination
    Vc_drug  = 1.0,      # mL       central volume
)

# Table S3: cellular injury (cisplatin)
CELL = dict(
    C_del            = 1.33e-3,  # min⁻¹         transit compartment rate
    k_injury         = 2.84e-3,  # min⁻¹/(ng/mL) injury rate
    k_recovery       = 1.22,     # min⁻¹         recovery rate
    k_death          = 1.42e-2,  # min⁻¹         max death rate
    IC50_Death       = 698,      # ng/mL          Cdeath at 50% death
    alpha_hill       = 1.9,      # -              Hill exponent
    k_regeneration   = 6.2e-3,   # min⁻¹         regeneration rate
    p_regen          = 1.26,     # -              regeneration power
    threshold_injury = 202,      # ng/mL          injury threshold
)

# Tables S2/S3: biomarker links
BM = dict(
    beta_aGST  = 52,       X0_aGST   = 0.00383,  # aGST
    beta_alb   = 1.5,      eta_alb0  = 0.995,
    Salb_mgml  = 45,       theta_alb = 0.00062,   # albumin
    beta_glu   = 3.8,      gamma_glu = 3.7,
    eta_S10    = 0.97,     eta_S30   = 0.03,
    Sglu_mgml  = 0.891,                           # glucose
    beta_Kim1  = 97,       tau_Kim1  = 4000,
    X0_Kim1    = 6e-6,                            # Kim-1
    beta_sod   = 0.95,     eta_PT_Na0= 0.65,      # Na reabsorption
)

# Table S2 + estimated: systems / sCr
SYS = dict(
    GFR0_mlmin = 2.648,    # mL/min   baseline GFR
    SCr0_mgdL  = 0.2,      # mg/dL    baseline serum creatinine
    BV_mL      = 16.25,    # mL       blood volume (250g × 6.5%)
    Emax_crea  = 0.9,      # -        max OCT2 inhibition by cisplatin
    EC50_crea  = 448,      # ng/mL    cisplatin EC50 for OCT2
    gamma_crea = 1.77,     # -        Hill exponent OCT2
    secr0      = 0.1,      # mL/min   baseline tubular secretion [ESTIMATED]
    S_TGF      = 0.20,     # -        GFR sensitivity to Na loss [ESTIMATED]
)

# Derived: creatinine synthesis calibrated to steady state
_SCr0_mgml = SYS['SCr0_mgdL'] / 100
_Cr_synth0 = (SYS['GFR0_mlmin'] + SYS['secr0']) * _SCr0_mgml

THETA = {**PK, **CELL, **BM, **SYS, 'Cr_synth0': _Cr_synth0}

# State variable index map
# 0:A0 1:A1 2:A2 3:C1 4:C2 5:C3 6:Cinjury 7:C5 8:Cdeath
# 9:Inj 10:Nec 11:Kim1tr 12:sCr_mg
_IDX = {n: i for i, n in enumerate(
    ['A0','A1','A2','C1','C2','C3','Cinjury','C5','Cdeath',
     'Inj','Nec','Kim1tr','sCr_mg']
)}

def odes(t, y, p):
    """ODE right-hand side. y: state vector (13), p: THETA dict."""
    (A0, A1, A2, C1, C2, C3, Cinjury, C5, Cdeath,
     Inj, Nec, Kim1tr, sCr_mg) = y

    # PK
    C_plasma  = A1 / p['Vc_drug']
    dA0       = -p['ka_drug'] * A0
    dA1       = (p['ka_drug'] * A0 - p['ke_drug'] * A1
                 - p['k12_drug'] * A1 + p['k21_drug'] * A2)
    dA2       = p['k12_drug'] * A1 - p['k21_drug'] * A2

    # Transit cascade
    dC1      = p['C_del'] * (C_plasma - C1)
    dC2      = p['C_del'] * (C1 - C2)
    dC3      = p['C_del'] * (C2 - C3)
    dCinjury = p['C_del'] * (C3 - Cinjury)
    dC5      = p['C_del'] * (Cinjury - C5)
    dCdeath  = p['C_del'] * (C5 - Cdeath)

    # Cellular injury
    Fcell         = max(0.0, 1.0 - Inj - Nec)
    RInjury       = p['k_injury'] * max(Cinjury - p['threshold_injury'], 0.0) * Fcell
    RRecovery     = p['k_recovery'] * Inj
    denom_death   = p['IC50_Death']**p['alpha_hill'] + Cdeath**p['alpha_hill']
    RDeath        = (p['k_death'] * Cdeath**p['alpha_hill'] / denom_death) * Inj
    RRegeneration = p['k_regeneration'] * max(Nec, 0.0)**p['p_regen']
    dInj          = RInjury - RRecovery - RDeath
    dNec          = RDeath - RRegeneration

    # Kim-1 transit
    E_Kim1   = p['beta_Kim1'] * (Inj + Nec)
    dKim1tr  = (E_Kim1 - Kim1tr) / p['tau_Kim1']

    # Serum creatinine
    sCr_mgml  = sCr_mg / p['BV_mL']
    E_OCT2    = (p['Emax_crea'] * C_plasma**p['gamma_crea']
                 / (p['EC50_crea']**p['gamma_crea'] + C_plasma**p['gamma_crea']))
    secr_eff  = p['secr0'] * (1.0 - E_OCT2)
    eta_PT_Na = max(0.0, p['eta_PT_Na0'] * (1.0 - p['beta_sod'] * (Inj + Nec)))
    delta_Na  = max(0.0, 1.0 - eta_PT_Na / p['eta_PT_Na0'])
    GFR_mult  = max(0.05, 1.0 - p['S_TGF'] * delta_Na)
    GFR       = p['GFR0_mlmin'] * GFR_mult
    Cr_clear  = (GFR + secr_eff) * sCr_mgml
    dsCr_mg   = p['Cr_synth0'] - Cr_clear

    return [dA0, dA1, dA2, dC1, dC2, dC3, dCinjury, dC5, dCdeath,
            dInj, dNec, dKim1tr, dsCr_mg]


def algebraics(t, y, p):
    """Compute all reported outputs from state vector."""
    (A0, A1, A2, C1, C2, C3, Cinjury, C5, Cdeath,
     Inj, Nec, Kim1tr, sCr_mg) = y

    C_plasma   = A1 / p['Vc_drug']
    Fcell      = max(0.0, 1.0 - Inj - Nec)
    sCr_mgml   = sCr_mg / p['BV_mL']
    sCr_mgdL   = sCr_mgml * 100.0
    fold_sCr   = sCr_mgdL / p['SCr0_mgdL']

    E_aGST     = 1.0 + p['beta_aGST'] * Nec
    fold_aGST  = E_aGST          # X0_aGST cancels

    fold_Kim1  = 1.0 + Kim1tr    # X0_Kim1 cancels

    E_OCT2     = (p['Emax_crea'] * C_plasma**p['gamma_crea']
                  / (p['EC50_crea']**p['gamma_crea'] + C_plasma**p['gamma_crea']))
    secr_eff   = p['secr0'] * (1.0 - E_OCT2)
    eta_PT_Na  = max(0.0, p['eta_PT_Na0'] * (1.0 - p['beta_sod'] * (Inj + Nec)))
    delta_Na   = max(0.0, 1.0 - eta_PT_Na / p['eta_PT_Na0'])
    GFR_mult   = max(0.05, 1.0 - p['S_TGF'] * delta_Na)
    GFR_mlmin  = p['GFR0_mlmin'] * GFR_mult

    return dict(
        C_plasma=C_plasma, Fcell=Fcell, frac_Inj=Inj, frac_Nec=Nec,
        sCr_mgml=sCr_mgml, sCr_mgdL=sCr_mgdL, fold_sCr=fold_sCr,
        fold_aGST=fold_aGST, fold_Kim1=fold_Kim1,
        GFR_mlmin=GFR_mlmin, Kim1tr=Kim1tr,
    )


def run_cisplatin(dose_mg_kg=2.5, bw_kg=0.25, p=None, t_end_days=22,
                  dt_min=60, method='LSODA'):
    """
    Simulate cisplatin IP dosing in rat.
    Returns dict with time array (minutes) and all output columns.
    """
    if p is None:
        p = THETA

    dose_ng  = dose_mg_kg * bw_kg * 1e6   # ng
    sCr0_mg  = p['SCr0_mgdL'] / 100 * p['BV_mL']

    # Initial conditions: A0 = dose, sCr_mg = baseline, everything else 0
    y0 = np.zeros(13)
    y0[0]  = dose_ng    # A0
    y0[12] = sCr0_mg    # sCr_mg

    t_end = t_end_days * 24 * 60   # minutes
    t_eval = np.arange(0, t_end + dt_min, dt_min)

    sol = solve_ivp(
        odes, [0, t_end], y0, args=(p,),
        method=method, t_eval=t_eval,
        rtol=1e-8, atol=1e-10, dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    # Build output dict
    t = sol.t
    Y = sol.y   # shape (13, n_times)

    out = dict(
        time   = t,
        t_days = t / (24 * 60),
        A0     = Y[0], A1=Y[1], A2=Y[2],
        C1     = Y[3], C2=Y[4], C3=Y[5],
        Cinjury= Y[6], C5=Y[7], Cdeath=Y[8],
        Inj    = Y[9], Nec=Y[10], Kim1tr=Y[11],
        sCr_mg = Y[12],
    )

    # Compute algebraic outputs at every time point
    alg_keys = list(algebraics(0, y0, p).keys())
    for k in alg_keys:
        out[k] = np.zeros(len(t))
    for i in range(len(t)):
        a = algebraics(t[i], Y[:, i], p)
        for k in alg_keys:
            out[k][i] = a[k]

    return out
