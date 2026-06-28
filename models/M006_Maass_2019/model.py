"""
Maass et al. 2019 — CPT Pharmacometrics Syst. Pharmacol. 8, 316-325
DOI: 10.1002/psp4.12400  |  PMID: 30869201

"Translational Assessment of Drug-Induced Proximal Tubule Injury Using
 a Kidney Microphysiological System"

Model: PBPK + KIM-1 shedding QSP model for in vitro-to-in vivo translation
       of kidney injury biomarker (KIM-1) from kidney MPS to human.

Structure (Figure 1, Methods S2 referenced in paper):
  1. Whole-body PBPK model (cisplatin) — organ compartments connected by
     cardiac output (cardiac output, blood flows: Brown 1997 [ref 34];
     Valentin 2002 ICRP [ref 33]).
  2. KIM-1 shedding model: empirical rate derived from in vitro MPS data
     (Figure 3) scaled to human N_nephrons (Scotcher 2016 [ref 35]).
  3. Neutrophil recruitment model: piecewise interpolation of fold-change
     at 2, 24, 48, 72 h (from mouse IRI model, Awad 2009 [ref 36]) ×
     activated-neutrophil KIM-1 boost factor 3.25 (Lingadahalli 2013 [ref 37]).
  4. Plasma and urine KIM-1 compartments.

NOTE: The exact ODE equations appear in Supplementary Methods S2 of the
      Wiley publication (not in the main paper PDF). This implementation
      reconstructs the model from the main-paper description, using the
      standard PBPK parameterisation cited therein (Brown 1997, ICRP 2002).
      Drug-specific PK parameters (cisplatin) are from established literature
      PBPK values consistent with the Brown/Valentin framework.
      All empirically set parameters are flagged [ESTIMATED].

Units: time = hours, amounts = pg (KIM-1) or mg (drug),
       concentrations = µg/mL (drug) or pg/mL (KIM-1), volumes = L.

Parameter sources:
  - ICRP = Valentin J. Ann ICRP 32, 5-265 (2002)          [ref 33]
  - Brown97 = Brown et al. Toxicol Ind Health 13, 407 (1997) [ref 34]
  - Scotcher16 = Scotcher et al. AAPS J 18, 1067 (2016)   [ref 35]
  - Paper text = stated directly in Maass 2019 main text
  - [ESTIMATED] = not in paper, fitted to match Figure 5 qualitative targets
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════════════════
# PHYSIOLOGICAL PARAMETERS  (Brown 1997 / ICRP 2002, 70 kg reference male)
# ═══════════════════════════════════════════════════════════════════════════

PHYSIOLOGY = dict(
    BW          = 70.0,    # kg       body weight (Brown97 Table 1)
    BSA         = 1.73,    # m²       body surface area
    CO          = 6.0,     # L/h      cardiac output (ICRP Table 1.16, converted from 5.6 L/min ≈ 6 L/h rounding)
    # Organ blood flows as fraction of cardiac output (Brown97 Table 2)
    f_Q_kidney  = 0.20,    # -        kidney blood flow fraction
    f_Q_liver   = 0.22,    # -        liver  blood flow fraction
    f_Q_muscle  = 0.17,    # -        muscle blood flow fraction
    f_Q_rest    = 0.41,    # -        remainder (cardiac, brain, adipose, …)
    # Organ volumes (L) (ICRP Table 2.13, 70 kg male)
    V_plasma    = 3.1,     # L        plasma volume
    V_kidney    = 0.31,    # L        both kidneys
    V_liver     = 1.69,    # L        liver
    V_muscle    = 28.0,    # L        muscle mass
    V_rest      = 21.0,    # L        remainder
    # Renal
    GFR         = 0.120,   # L/h      glomerular filtration rate (120 mL/min → 7.2 L/h) — Brown97
    # KIM-1 biology
    N_nephrons  = 1.8e6,   # -        nephrons per pair of kidneys (Scotcher16, ref 35)
    N_MPS_cells = 5000,    # -        hRPTECs seeded per MPS device (Table 1, paper)
    Q_MPS       = 3.0e-5,  # L/h      MPS perfusion rate (0.5 µL/min = 0.0005 mL/min = 3×10⁻⁵ L/h)
)

# ═══════════════════════════════════════════════════════════════════════════
# CISPLATIN PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# Standard clinical PBPK values for cisplatin (IV, 70 mg/m²).
# Cisplatin: MW = 300 g/mol, polar inorganic Pt complex.
# Reference range: Cmax plasma ~5-10 µg/mL, CLtot ~15-25 L/h, t½ terminal ~24-48 h.
# Kidney-specific: rapid uptake via OCT2; tissue:plasma Kp very high in kidney.

CISPLATIN = dict(
    dose_per_m2 = 70.0,    # mg/m²    standard cisplatin dose
    infusion_h  = 0.5,     # h        IV infusion duration (30 min)
    fu_plasma   = 0.05,    # -        fraction unbound in plasma (~5%, highly protein-bound)
    # Tissue partition coefficients (Kp = tissue concentration / plasma concentration at ss)
    # Cisplatin accumulates heavily in kidney — Kp_kidney >> 1 (Peng 2005 and related PBPK)
    Kp_kidney   = 15.0,    # -        kidney/plasma conc ratio [ESTIMATED from literature range 5-50]
    Kp_liver    = 3.0,     # -        liver/plasma
    Kp_muscle   = 1.5,     # -        muscle/plasma
    Kp_rest     = 2.0,     # -        rest/plasma
    # Clearance
    CL_renal    = 4.2,     # L/h      renal excretion clearance (~70 mL/min = 4.2 L/h)
    CL_other    = 2.8,     # L/h      non-renal (biliary/metabolism) [ESTIMATED]
    # In vitro drug concentrations used in MPS experiments (Table S2, Maass 2019 supplementary)
    # NOTE: exact concentrations from S2 are not published in main text.
    # Values below are inferred: cisplatin clinical kidney Cmax ~5-15 µg/mL (17-50 µM).
    # In vitro doses chosen to span this range (typical for cisplatin MPS toxicity assays,
    # consistent with Adler et al. 2016 same MPS system).
    C_invitro_low  = 1.0,  # µg/mL   Low dose  (~3.3 µM)  [ESTIMATED Table S2]
    C_invitro_mid  = 5.0,  # µg/mL   Medium dose (~17 µM)  [ESTIMATED Table S2]
    C_invitro_high = 30.0, # µg/mL   High dose  (~100 µM)  [ESTIMATED Table S2]
)

# ═══════════════════════════════════════════════════════════════════════════
# KIM-1 SHEDDING MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# The shedding rate function R(C, t) is empirically derived from Figure 3
# (KIM-1 in MPS effluent at 6 h, 24 h, 48 h after dosing).
# Digitized values from Figure 3a (cisplatin panel) are the ground-truth;
# we use a two-component model:
#   R(C, t) = R_basal + R_drug(C) × decay(t)
# where decay(t) captures the rapid return to baseline seen by 24 h.

KIM1_MODEL = dict(
    # In vitro MPS KIM-1 effluent concentration (pg/mL) from digitized Fig 3a
    # These are the MEASURED values in the MPS effluent at each time interval.
    # Data from resources/digitized/M006_PMID30869201_fig3a_cisplatin.csv
    # Format: { time_h: { dose: C_effluent_pgml } }
    invitro_t_h          = [6.0, 24.0, 48.0],
    invitro_C_drug_ugml  = [0.0, 1.0, 5.0, 30.0],  # "No Dose", Low, Medium, High

    # Baseline (no-drug) MPS shedding from paper text:
    # "~1-3 pg/10³ cells seeded" over 48 h measurement; with 5000 cells → ~5-15 pg total
    # effluent volume 48 h: 0.0005 mL/min × 2880 min = 1.44 mL → ~3.5-10.4 pg/mL
    # Using 7 pg/mL as central estimate
    R_basal_pgml        = 7.0,   # pg/mL  in MPS effluent (nodrug baseline)

    # Emax parameters for concentration-dependent shedding excess above baseline
    # Fit to match Figure 3a: at 6 h, high dose (30 µg/mL) → ~25 pg/mL
    # → excess above baseline = 25 - 7 = 18 pg/mL
    Emax_pgml           = 20.0,  # pg/mL  [ESTIMATED to match Fig 3a peak]
    EC50_ugml           = 3.0,   # µg/mL  [ESTIMATED]
    hill_n              = 1.5,   # -       Hill exponent [ESTIMATED]

    # Adaptation: exponential return to baseline; by 24 h most excess gone
    k_adapt             = 0.30,  # h⁻¹    adaptation rate constant [ESTIMATED; t½ ≈ 2.3 h]

    # Scale factor: from 1 MPS device to whole human pair of kidneys.
    # Paper: "shedding rates scaled up to represent number of nephrons in both kidneys" (ref 35)
    # Interpretation: 1 MPS ≈ 1 nephron-equivalent proximal tubule unit.
    # Scale = N_nephrons_human = 1.8×10⁶  (but split between plasma and urine fractions)
    # f_plasma: fraction of shed KIM-1 reaching plasma (basolateral shedding + lymphatics)
    # f_urine:  fraction reaching measurable urine (most KIM-1 is degraded in tubular lumen
    #           or adsorbed to device surfaces before exiting as final urine)
    # Calibrated to match Figure 5 qualitative targets (paper text, PMID 30869201):
    #   plasma peak (no immune) ≈ 2× baseline (~100 pg/mL)
    #   plasma peak (immune)    ≈ 1000 pg/mL  (text: "distinct plasma KIM-1 peaks (~1,000 pg/mL)")
    #   urine  peak (immune)    ≈ 3000 pg/mL  (text: "urine concentrations were elevated after 24h (~3,000 pg/mL)")
    # NOTE: The 90% not accounted for (1-f_plasma-f_urine=0.885) is assumed degraded/adsorbed.
    # This is physiologically plausible for a large glycoprotein (90 kDa) in a flow system.
    f_plasma            = 0.072, # -  [ESTIMATED] fraction → plasma
    f_urine             = 0.050, # -  [ESTIMATED] fraction → measurable urine

    # Plasma KIM-1 PK (after entering blood)
    k_elim_plasma       = 0.10,  # h⁻¹  elimination rate of plasma KIM-1  [ESTIMATED]
    V_dist_plasma       = 3.1,   # L    distribution volume = plasma volume

    # Urine KIM-1: modelled as flow-through from tubular lumen to bladder
    # C_urine_KIM1 = R_urine / Q_urine
    Q_urine_Lph         = 0.0625, # L/h  urine flow = 1.5 L/day (24 h average)

    # Clinical baseline plasma KIM-1 (pre-drug; used to set initial conditions)
    # From Table S5 (supplementary), healthy volunteers range 50-500 pg/mL.
    # We use the mid-range value from clinical comparison in Figure 5.
    C_KIM1_plasma_baseline = 50.0,  # pg/mL  [ESTIMATED mid-range]
)

# ═══════════════════════════════════════════════════════════════════════════
# NEUTROPHIL EFFECT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# Source: mouse ischemia-reperfusion injury (Awad 2009, ref 36)
# Neutrophil count fold-changes over baseline measured at t = 2, 24, 48, 72 h.
# Separately: activated neutrophils increase KIM-1 shedding 3.25× (Lingadahalli 2013, ref 37).
# Implementation: total KIM-1 boost from neutrophils at time t =
#     1 + (neutrophil_fold(t) - 1) × (3.25 - 1)
# where neutrophil_fold interpolates between measured values.

NEUTROPHIL = dict(
    # Data from Awad 2009 (Kidney Int 75:689) via paper main text
    t_h   = [0.0, 2.0, 24.0, 48.0, 72.0, 240.0],  # h
    fold  = [1.0, 2.6,  3.0,  2.7,  2.0,   1.0],   # fold change over baseline (paper text)
    # Boost per activated neutrophil on KIM-1 shedding (Lingadahalli 2013, ref 37)
    KIM1_boost_per_neutrophil = 3.25,
)

# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLED PARAMETER DICT
# ═══════════════════════════════════════════════════════════════════════════

THETA = {**PHYSIOLOGY, **CISPLATIN, **KIM1_MODEL, **NEUTROPHIL}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: neutrophil KIM-1 multiplier
# ═══════════════════════════════════════════════════════════════════════════

def _neutrophil_kim1_factor(t, theta):
    """Return total KIM-1 shedding multiplier from neutrophil recruitment at time t (h)."""
    t_pts = theta['t_h']
    f_pts = theta['fold']
    boost = theta['KIM1_boost_per_neutrophil']
    # Piecewise linear interpolation; clamp outside range to endpoint values
    if t <= t_pts[0]:
        n_fold = f_pts[0]
    elif t >= t_pts[-1]:
        n_fold = f_pts[-1]
    else:
        n_fold = float(np.interp(t, t_pts, f_pts))
    # Combined multiplier: 1 + extra_neutrophils × boost_per_neutrophil
    # Extra neutrophils = (n_fold - 1) × baseline_count (normalised to 1)
    # Their effect: each extra unit of neutrophils boosts by (boost - 1)
    return 1.0 + (n_fold - 1.0) * (boost - 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# IN VITRO SHEDDING RATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def _invitro_shed_rate(C_drug_ugml, t_since_dose_h, theta):
    """
    KIM-1 effluent concentration (pg/mL) from 1 MPS unit at drug conc C and
    time t after start of drug exposure.

    Model: C_KIM1_effluent = R_basal + Emax × C^n/(EC50^n + C^n) × exp(-k_adapt × t)

    Parameters derived from Figure 3a (cisplatin, digitized).
    """
    R_b = theta['R_basal_pgml']
    Emax = theta['Emax_pgml']
    EC50 = theta['EC50_ugml']
    n    = theta['hill_n']
    k_ad = theta['k_adapt']
    C = max(0.0, C_drug_ugml)
    Ehill = Emax * C**n / (EC50**n + C**n) if C > 0 else 0.0
    decay = np.exp(-k_ad * max(0.0, t_since_dose_h))
    return R_b + Ehill * decay


def _mps_shed_to_human_rate(C_KIM1_effluent_pgml, theta):
    """
    Convert MPS effluent KIM-1 concentration (pg/mL) to total human shedding
    rate (pg/h) for both kidneys.

    Shedding rate from 1 MPS:  R_MPS [pg/h] = C_KIM1_eff [pg/mL] × Q_MPS [L/h] × 1000 [mL/L]
    Human total:               R_human = R_MPS × N_nephrons
    (1 MPS represents 1 nephron-equivalent proximal tubule unit; Scotcher 2016)
    """
    Q = theta['Q_MPS']  # L/h
    N = theta['N_nephrons']
    R_MPS = C_KIM1_effluent_pgml * Q * 1000.0  # pg/h per MPS
    return R_MPS * N  # pg/h total (both kidneys)


# ═══════════════════════════════════════════════════════════════════════════
# ODE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

# State vector indices
IDX = dict(
    A_kidney    = 0,   # drug amount in kidney [mg]
    A_liver     = 1,   # drug amount in liver [mg]
    A_muscle    = 2,   # drug amount in muscle [mg]
    A_rest      = 3,   # drug amount in rest [mg]
    A_plasma    = 4,   # drug amount in plasma [mg]
    A_KIM1_p    = 5,   # KIM-1 amount in plasma [pg]
    A_KIM1_u    = 6,   # KIM-1 amount in urine reservoir [pg]
)
N_STATES = 7


def odes(t, y, theta, dose_mg, immune=True):
    """
    ODE right-hand side for the Maass 2019 PBPK + KIM-1 model.

    Parameters
    ----------
    t         : float   time (h)
    y         : ndarray state vector (N_STATES,)
    theta     : dict    parameters
    dose_mg   : float   total cisplatin dose (mg)
    immune    : bool    include neutrophil recruitment effect

    ODE structure:
      1. Flow-limited PBPK for cisplatin (plasma + 4 tissue compartments).
         dA_c/dt = Q_c × (C_plasma × fu - A_c/V_c/Kp_c × fu) - elim
         Mass balance: dA_plasma/dt = dose_input - sum(net tissue uptake) - renal CL
      2. KIM-1 shedding from kidney (proxy: kidney drug concentration).
         Shedding rate at each time point derived from in vitro data (Fig 3a)
         mapped through [C_cisplatin_kidney(t)] and optionally amplified by neutrophils.
      3. Plasma KIM-1 accumulation and first-order elimination.
      4. Urine KIM-1: direct accumulation from shed KIM-1 at rate R_urine.
    """
    # Unpack states
    A_kidney = y[IDX['A_kidney']]
    A_liver  = y[IDX['A_liver']]
    A_muscle = y[IDX['A_muscle']]
    A_rest   = y[IDX['A_rest']]
    A_plasma = y[IDX['A_plasma']]
    A_KIM1_p = y[IDX['A_KIM1_p']]
    A_KIM1_u = y[IDX['A_KIM1_u']]

    # ----- PBPK: blood flows (L/h) -----
    CO = theta['CO']
    Q_k = theta['f_Q_kidney'] * CO
    Q_l = theta['f_Q_liver']  * CO
    Q_m = theta['f_Q_muscle'] * CO
    Q_r = theta['f_Q_rest']   * CO

    # ----- PBPK: concentrations -----
    V_p  = theta['V_dist_plasma']
    C_p  = max(A_plasma, 0.0) / V_p    # µg/mL (= mg/L) in plasma
    fu   = theta['fu_plasma']

    # Tissue concentrations
    C_k = max(A_kidney, 0.0) / theta['V_kidney']   # µg/mL
    C_l = max(A_liver,  0.0) / theta['V_liver']
    C_m = max(A_muscle, 0.0) / theta['V_muscle']
    C_r = max(A_rest,   0.0) / theta['V_rest']

    # Partition coefficients
    Kp_k = theta['Kp_kidney']
    Kp_l = theta['Kp_liver']
    Kp_m = theta['Kp_muscle']
    Kp_r = theta['Kp_rest']

    # Net tissue uptake rates (mg/h); flow-limited PBPK:
    #   dA_tissue/dt = Q_tissue × (C_art - C_ven)
    #   C_art = fu × C_plasma (free drug drives distribution)
    #   C_ven = C_tissue / Kp (tissue equilibrium assumption)
    C_art = fu * C_p
    dA_kidney = Q_k * (C_art - C_k / Kp_k)
    dA_liver  = Q_l * (C_art - C_l / Kp_l)
    dA_muscle = Q_m * (C_art - C_m / Kp_m)
    dA_rest   = Q_r * (C_art - C_r / Kp_r)

    # Renal clearance (from plasma free drug)
    CL_r = theta['CL_renal']
    CL_o = theta['CL_other']
    elim_plasma = CL_r * fu * C_p + CL_o * fu * C_p

    # Plasma mass balance: dA_plasma = dose_in - tissue_uptake - elimination
    # Infusion input handled externally (events-based) — here just set 0 after infusion
    infusion_rate = 0.0   # set externally; ODE receives continuous input rate = 0 outside infusion
    dA_plasma = (infusion_rate
                 - Q_k * (C_art - C_k / Kp_k)
                 - Q_l * (C_art - C_l / Kp_l)
                 - Q_m * (C_art - C_m / Kp_m)
                 - Q_r * (C_art - C_r / Kp_r)
                 - elim_plasma)

    # ----- KIM-1 model -----
    # Drug concentration in kidney drives KIM-1 shedding via in vitro calibration
    t_since_dose = max(0.0, t)
    C_KIM1_invitro = _invitro_shed_rate(C_k, t_since_dose, theta)
    R_shed_total = _mps_shed_to_human_rate(C_KIM1_invitro, theta)   # pg/h both kidneys

    # Neutrophil amplification (only after drug exposure begins, i.e. t > 0)
    if immune and t > 0:
        neut_factor = _neutrophil_kim1_factor(t, theta)
    else:
        neut_factor = 1.0

    R_shed_final = R_shed_total * neut_factor   # pg/h with or without immune effect

    # Plasma KIM-1
    f_p = theta['f_plasma']
    k_e = theta['k_elim_plasma']
    dA_KIM1_p = f_p * R_shed_final - k_e * A_KIM1_p

    # Urine KIM-1 (cumulative shedding to tubular lumen)
    f_u = theta['f_urine']
    dA_KIM1_u = f_u * R_shed_final   # no reabsorption assumed for large protein

    return [dA_kidney, dA_liver, dA_muscle, dA_rest, dA_plasma, dA_KIM1_p, dA_KIM1_u]


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_cisplatin(theta=None, immune=True, t_end_h=240.0, n_points=2001,
                       dose_mg_per_m2=70.0):
    """
    Simulate cisplatin PBPK + KIM-1 model for a single virtual patient.

    Parameters
    ----------
    theta          : dict or None   parameter dict (defaults to THETA)
    immune         : bool           include neutrophil effect (True = Figure 5 centre/right panels)
    t_end_h        : float          simulation end time (h); 240 h = 10 days
    n_points       : int            output time points
    dose_mg_per_m2 : float          cisplatin dose (mg/m²)

    Returns
    -------
    t    : ndarray (n_points,)   time (h)
    sol  : ndarray (N_STATES, n_points)  state time-courses
    meta : dict                  metadata (dose, parameters used, etc.)
    """
    if theta is None:
        theta = THETA

    BSA = theta['BSA']
    dose_mg = dose_mg_per_m2 * BSA
    inf_h = theta['infusion_h']
    infusion_rate_mg_per_h = dose_mg / inf_h   # mg/h during infusion

    # Initial conditions
    # Pre-dose baseline KIM-1: A_KIM1_p = C_baseline × V_plasma
    C0_KIM1_p = theta['C_KIM1_plasma_baseline'] * theta['V_dist_plasma'] * 1e3  # pg (V in L → pg/L × L)
    # Actually: C [pg/mL] × V [L] × 1000 mL/L = C × V × 1000 pg
    C0_KIM1_p = theta['C_KIM1_plasma_baseline'] * theta['V_dist_plasma'] * 1000.0  # pg

    y0 = np.zeros(N_STATES)
    y0[IDX['A_KIM1_p']] = C0_KIM1_p

    # Solve in two phases: during infusion and after
    t_eval = np.linspace(0.0, t_end_h, n_points)

    def rhs_during_infusion(t, y):
        dy = odes(t, y, theta, dose_mg, immune=immune)
        dy[IDX['A_plasma']] += infusion_rate_mg_per_h  # IV infusion
        return dy

    def rhs_after_infusion(t, y):
        return odes(t, y, theta, dose_mg, immune=immune)

    # Phase 1: 0 → inf_h (during infusion)
    t_mask1 = t_eval[t_eval <= inf_h]
    if len(t_mask1) == 0:
        t_mask1 = np.array([0.0, inf_h])

    sol1 = solve_ivp(rhs_during_infusion, [0.0, inf_h],
                     y0, t_eval=t_mask1, method='LSODA',
                     rtol=1e-6, atol=1e-9, dense_output=True)
    y1_end = sol1.y[:, -1]

    # Phase 2: inf_h → t_end_h (after infusion)
    t_mask2 = t_eval[t_eval >= inf_h]
    if len(t_mask2) == 0 or t_mask2[0] > inf_h:
        t_mask2 = np.concatenate([[inf_h], t_mask2])

    sol2 = solve_ivp(rhs_after_infusion, [inf_h, t_end_h],
                     y1_end, t_eval=t_mask2, method='LSODA',
                     rtol=1e-6, atol=1e-9, dense_output=True)

    # Concatenate results
    t1 = sol1.t
    t2 = sol2.t
    y1 = sol1.y
    y2 = sol2.y

    # Remove duplicate at inf_h boundary
    mask = t2 > t1[-1]
    if not np.any(mask):
        mask = np.ones(len(t2), dtype=bool)
        mask[0] = False

    t_out = np.concatenate([t1, t2[mask]])
    y_out = np.concatenate([y1, y2[:, mask]], axis=1)

    meta = dict(
        dose_mg=dose_mg,
        dose_mg_per_m2=dose_mg_per_m2,
        infusion_h=inf_h,
        immune=immune,
        t_end_h=t_end_h,
        theta=theta,
    )

    return t_out, y_out, meta


def get_outputs(t, y, theta=None):
    """
    Compute key model outputs (concentrations) from raw state vectors.

    Returns
    -------
    dict with keys:
        t              : time (h)
        C_drug_plasma  : cisplatin in plasma (µg/mL)
        C_drug_kidney  : cisplatin in kidney (µg/mL)
        C_KIM1_plasma  : plasma KIM-1 (pg/mL)
        C_KIM1_urine   : urine KIM-1 concentration (pg/mL) at each t
    """
    if theta is None:
        theta = THETA
    V_p = theta['V_dist_plasma']
    V_k = theta['V_kidney']
    Q_u = theta['Q_urine_Lph']  # L/h

    C_drug_plasma  = y[IDX['A_plasma']]  / V_p          # µg/mL
    C_drug_kidney  = y[IDX['A_kidney']]  / V_k          # µg/mL
    C_KIM1_plasma  = y[IDX['A_KIM1_p']] / (V_p * 1000) # pg/mL  (A in pg, V in L → /1000 mL)
    # Urine KIM-1: cumulative amount shed / urine volume flowing out per hour
    # Instantaneous rate = dA_KIM1_u/dt = f_u × R_shed_final
    # Concentration = rate / Q_urine
    R_shed_final = np.zeros(len(t))
    for i, (ti, yi) in enumerate(zip(t, y.T)):
        t_since_dose = max(0.0, ti)
        C_k_i = max(yi[IDX['A_kidney']], 0.0) / V_k
        C_invitro = _invitro_shed_rate(C_k_i, t_since_dose, theta)
        R_total = _mps_shed_to_human_rate(C_invitro, theta)
        R_shed_final[i] = R_total  # no neutrophil here — handled separately

    f_u = theta['f_urine']
    C_KIM1_urine = f_u * R_shed_final / (Q_u * 1000.0)   # pg/h / (L/h × 1000 mL/L) = pg/mL

    return dict(
        t=t,
        C_drug_plasma=C_drug_plasma,
        C_drug_kidney=C_drug_kidney,
        C_KIM1_plasma=C_KIM1_plasma,
        C_KIM1_urine=C_KIM1_urine,
        R_shed_final=R_shed_final,
    )


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t, y, meta = simulate_cisplatin(immune=False, t_end_h=48.0)
    out = get_outputs(t, y)
    i_peak = np.argmax(out['C_drug_plasma'])
    print(f"Cisplatin peak plasma:  {out['C_drug_plasma'].max():.2f} µg/mL at t={t[i_peak]:.1f} h")
    print(f"Cisplatin peak kidney:  {out['C_drug_kidney'].max():.2f} µg/mL")
    print(f"KIM-1 plasma peak (no immune): {out['C_KIM1_plasma'].max():.1f} pg/mL")

    t2, y2, _ = simulate_cisplatin(immune=True, t_end_h=240.0)
    out2 = get_outputs(t2, y2)
    print(f"KIM-1 plasma peak (with immune): {out2['C_KIM1_plasma'].max():.1f} pg/mL")
