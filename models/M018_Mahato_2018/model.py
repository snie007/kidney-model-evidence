"""
Mahato et al. 2018 — DKD Glomerular Hypertension Model (Mouse db/db)
Paper: "Mathematical model of hemodynamic mechanisms and consequences of
        glomerular hypertension in diabetic mice"
PMID: 30564457  PMCID: PMC6288095
NPJ Syst Biol Appl 2018;4:41

Extends the Hallow & Gebremichael 2017 (M003) renal QSP model with:
  — Mouse (db/db) parameterisation
  — SGLT2-mediated glucose-sodium coupling in proximal tubule
  — Adaptive afferent arteriole dilation driven by hyperglycaemia (ΔDaa ODE)
  — Glomerular hypertrophy / surface-area increase (ΔSA ODE, τ = 750 days)
  — Glomerulosclerosis / permeability loss (ΔPerm ODE, τ = 40 000 days)
  — Nephron loss (ΔNephrons ODE, τ = 40 000 days — ESTIMATED same as ΔPerm)
  — Albuminuria sieving coefficient (ΔKalb ODE)

State variables (27 total):
  0 – 21  Same 22 states as M003 Hallow 2017 (see M003/model.py)
  22      delta_Daa       fractional afferent arteriole dilation  (0 → Daa_max)
  23      delta_SA        fractional glomerular surface area gain  (0 → SA_max)
  24      delta_Perm      fractional permeability loss (glomerulosclerosis) (0 → 1)
  25      delta_Nephrons  fractional nephron loss (0 → 1; N_func = N0 × (1 − ΔN))
  26      delta_Kalbumin  sieving coefficient gain (0 → unbounded; adds to Kalb0)

ESTIMATED parameters (not directly stated in paper — flagged inline):
  • nom_mean_filling_pressure (3.23 mmHg) — scaled from mouse CO/MAP target
  • nom_systemic_arterial_resistance — computed to give MAP = 98 mmHg at CO = 0.088
  • tau_daa (4e6 min·dL/mg) — units inferred so effective τ ≈ 7–10 days at ΔBG = 410 mg/dL
  • tau_nephron (57,600,000 min) — assumed equal to tau_perm (not stated in paper)
  • tau_albumin (57,600,000 min) — assumed equal to tau_perm (not stated in paper)
  • Blood glucose profile — sigmoid ramp (not given explicitly in paper)

Time unit: MINUTES (matching M003 base model).
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable 1 / (1 + exp(x))."""
    return 1.0 / (1.0 + math.exp(max(-500.0, min(500.0, x))))


# ---------------------------------------------------------------------------
# Blood glucose input profile
# ---------------------------------------------------------------------------

MIN_PER_WEEK = 7.0 * 24.0 * 60.0  # 10 080 min/week

def blood_glucose_mg_dl(t_min, scenario="normal"):
    """
    Time-dependent blood glucose profile (mg/dL).

    scenario = "normal"  : lean control db/m, BG ≈ 90–130 mg/dL
    scenario = "dbdb"    : diabetic db/db, BG rises sigmodally to ~500 mg/dL
    scenario = "dbdb_unx": after uninephrectomy at week 8–10 (same BG trajectory
                           as db/db — model distinguishes via renal mass)

    The exact BG trajectory is not specified analytically in the paper (it is
    derived from experimental data). This sigmoid approximation is ESTIMATED
    to match Fig 2A of Mahato et al. 2018.
    """
    if scenario == "normal":
        # C57BL/6J lean control mice: fasting BG ≈ 90 mg/dL (= BG_nom reference).
        # Using 90.0 ensures BG_excess = 0 → no afferent dilation (delta_Daa stays at 0).
        # This matches the normoglycaemic reference used in Mahato 2018 Table 1.
        return 90.0  # mg/dL — normoglycaemic C57BL/6J fasting reference

    # db/db: sigmoid ramp from ~130 mg/dL to ~500 mg/dL over ~8 weeks
    t_weeks = t_min / MIN_PER_WEEK
    BG_0   = 130.0   # mg/dL initial (slightly elevated even at week 0 in db/db)
    BG_max = 500.0   # mg/dL plateau (ESTIMATED — from Fig 2A scatter ~400–600)
    t_half = 6.0     # weeks to half-max (ESTIMATED)
    k      = 0.5     # sigmoid steepness (ESTIMATED)
    BG = BG_0 + (BG_max - BG_0) / (1.0 + math.exp(-k * (t_weeks - t_half)))
    return BG


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def make_params(scenario="normal"):
    """
    Return mouse-parameterised dict for M018.

    Table 1 values (Mahato et al. 2018, Table 1) override human M003 defaults.
    All ESTIMATED parameters are labelled with # ESTIMATED.
    """
    p = {}

    # ── Unit-conversion constants ──────────────────────────────────────────
    p["nL_mL"]               = 1e6
    p["dl_ml"]               = 0.01
    p["L_dL"]                = 10.0
    p["L_mL"]                = 1000.0
    p["L_m3"]                = 0.001
    p["g_mg"]                = 0.001
    p["ng_mg"]               = 1e-6
    p["min_day"]             = 1440.0
    p["min_week"]            = MIN_PER_WEEK
    p["MW_creatinine"]       = 113.12
    p["MW_glucose"]          = 180.0   # g/mol
    p["pi"]                  = 3.14159
    p["viscosity_length_constant"] = 1.5e-9
    p["gamma"]               = 1.16667e-5
    p["mmHg_Nperm2_conv"]    = 133.32

    # ── Species-scaling (mouse values) ────────────────────────────────────
    p["ECF_scale_species"]          = 1.0
    p["BV_scale_species"]           = 1.0
    p["water_intake_species_scale"] = 1.0
    p["CO_scale_species"]           = 1.0

    # ── Mouse systemic parameters (Table 1 + physiology) ──────────────────
    p["nominal_map_setpoint"]   = 98.0        # mmHg  — Table 2 Mahato 2018
    p["CO_nom"]                 = 0.088       # L/min — Table 2 Mahato 2018
    p["ECF_nom"]                = 0.006       # L     — ~20% × 30g body weight
    p["blood_volume_nom"]       = 0.045       # L     — ~1.5 mL/g × 30g mouse
    p["Na_intake_rate"]         = 0.576 / 1440.0  # mEq/min  (0.576 mEq/day, Table 1)
    p["nom_water_intake"]       = 0.005       # L/day — Table 1 Mahato 2018
    p["ref_Na_concentration"]   = 140.0       # mEq/L — Table 1 (same as human)
    p["plasma_protein_concentration"] = 3.4   # g/dL  — Table 1 Mahato 2018
    p["equilibrium_serum_creatinine"] = 0.2   # mg/dL — ESTIMATED mouse baseline
    p["P_venous"]               = 4.0         # mmHg  — same as M003 human
    p["R_venous"]               = 3.4         # mmHg  — same as M003 human (ESTIMATED)
    p["nom_right_atrial_pressure"] = 0.87     # mmHg
    p["venous_compliance"]      = 0.13        # same as human (ESTIMATED)

    # Nominal mouse filling pressure (ESTIMATED — computed to give CO_nom at MAP_nom)
    # CO = nom_mfp / R_tvr;  R_tvr = (8*R_venous + SAR) / 31
    # SAR = MAP/CO - R_venous;  R_tvr = (8*3.4 + 1110.2)/31 = 36.69
    p["nom_mean_filling_pressure"] = 3.23     # mmHg  ESTIMATED

    # ── Renal parameters (Table 1 Mahato 2018) ────────────────────────────
    p["nom_renal_blood_flow_L_min"] = 0.0018  # L/min (1.8 mL/min, Table 2)
    p["baseline_nephrons"]      = 10_000      # Table 1 Mahato 2018
    p["nom_Kf"]                 = 3.9         # nL/min/mmHg — Table 1 (same as human)
    p["nom_oncotic_pressure_difference"] = 10.17  # mmHg ESTIMATED for mouse plasma protein
    p["P_renal_vein"]           = 4.0

    # Arteriole diameters (Table 1 Mahato 2018)
    p["nom_afferent_diameter"]  = 14.0e-6    # m  (14 µm, Table 1)
    p["nom_efferent_diameter"]  = 10.5e-6    # m  (10.5 µm, Table 1)

    # Preafferent resistance — ESTIMATED to give Pgc = 37.5 mmHg, RBF = 0.0018 L/min
    # Computed: preaff_R = (MAP - Pgc - RBF × R_aff/N) / RBF
    #                     = (98 - Pgc_target - 7.04) / 0.0018
    # With Pgc_target = 37.5 → preaff_R ≈ 29 722
    p["nom_preafferent_arteriole_resistance"] = 29_722.0  # mmHg/(L/min) ESTIMATED

    # Tubule geometry (Table 1 Mahato 2018 — mouse PT total 2.2 mm)
    p["Dc_pt_nom"]   = 27.0e-6    # m — same as human
    p["Dc_lh"]       = 17.0e-6    # m — same as human (ESTIMATED)
    p["Dc_dt"]       = 17.0e-6    # m — same as human (ESTIMATED)
    p["Dc_cd"]       = 22.0e-6    # m — same as human (ESTIMATED)
    p["L_pt_s1_nom"] = 0.0022 / 3.0  # m  (2.2 mm / 3 segments, Table 1)
    p["L_pt_s2_nom"] = 0.0022 / 3.0  # m
    p["L_pt_s3_nom"] = 0.0022 / 3.0  # m
    p["L_lh_des"]    = 0.003     # m — ESTIMATED (scaled from human 10 mm)
    p["L_lh_asc"]    = 0.003     # m — ESTIMATED
    p["L_dct"]       = 0.0015    # m — ESTIMATED
    p["L_cd"]        = 0.003     # m — ESTIMATED
    p["tubular_compliance"] = 0.2

    # Tubular pressures (mmHg) — ESTIMATED (same as human, scaled mildly)
    p["Pc_pt_mmHg"]     = 14.0
    p["Pc_lh_des_mmHg"] = 10.5
    p["Pc_lh_asc_mmHg"] = 7.0
    p["Pc_dt_mmHg"]     = 3.0
    p["Pc_cd_mmHg"]     = 2.0
    p["P_interstitial_mmHg"] = 5.0

    # ── Mouse Na reabsorption (Table 1 Mahato 2018) ───────────────────────
    p["eta_pt_nonSGLT2"]             = 0.76   # fractional PT Na reabsorb (non-SGLT2, Table 1)
    p["nominal_loh_na_reabsorption"] = 0.80   # same as human M003 (ESTIMATED)
    p["nominal_dt_na_reabsorption"]  = 0.50   # same as human M003 (ESTIMATED)
    p["LoH_flow_dependence"]         = 1.0

    # ── SGLT2 parameters (Table 1 Mahato 2018) ────────────────────────────
    p["RTG_normal"]     = 180.0   # mg/dL — renal threshold glucose (normal)
    p["RTG_diabetic"]   = 400.0   # mg/dL — renal threshold in db/db (Table 1 / text)
    p["BG_nom"]         = 90.0    # mg/dL — normoglycaemic reference

    # ── Albuminuria parameters (Table 1 Mahato 2018) ──────────────────────
    p["Kalbumin0"]      = 0.0006  # dimensionless sieving coefficient (Table 1)
    p["eta_albumin"]    = 0.99    # fractional tubular albumin reabsorption (ESTIMATED)
    p["RC_albumin"]     = 2.5e-9  # mg/min per nephron  (Table 1: 2.5×10⁻⁶ mg/min total /10000 neph)
    p["C_albumin"]      = 4000.0  # mg/L albumin in plasma (ESTIMATED ~4 g/dL)

    # ── DKD extension — slow ODE parameters ──────────────────────────────
    # Afferent arteriole dilation (Eq 19 Mahato 2018)
    p["delta_Daa_max"]  = 0.25    # max fractional dilation (25%, Table 1)
    # tau_daa = 4e6 (Table 1); interpreted as min·dL/mg so effective τ ≈ 9756 min
    # at ΔBG = 410 mg/dL — ESTIMATED units; gives ~7 day time constant
    p["tau_daa"]        = 4.0e6   # min·dL/mg  — Table 1 (units ESTIMATED)

    # Glomerular surface area / hypertrophy (Eq 15 Mahato 2018)
    p["delta_SA_max"]   = 0.50    # max fractional SA increase (50%, Table 1)
    p["tau_SA"]         = 750.0 * 1440.0   # min (750 days × 1440 min/day, Table 1)

    # Glomerulosclerosis / permeability loss (Eq 16 Mahato 2018)
    p["delta_Perm_max"] = 1.00    # max fractional permeability loss (100%, Table 1)
    p["tau_perm"]       = 40_000.0 * 1440.0  # min (40 000 days, Table 1)

    # Nephron loss (ESTIMATED — coupled to glomerulosclerosis; τ = τperm)
    p["tau_nephron"]    = 40_000.0 * 1440.0  # min  ESTIMATED

    # Albumin sieving coefficient change (ESTIMATED — τ = τperm)
    p["tau_albumin"]    = 40_000.0 * 1440.0  # min  ESTIMATED
    p["delta_Kalbumin_max"] = 0.10  # max additional sieving (ESTIMATED)

    # ── RAAS (same as M003 human, nominal values) ─────────────────────────
    p["concentration_to_renin_activity_conversion_plasma"] = 61.0
    p["nominal_equilibrium_PRA"]    = 1000.0
    p["nominal_equilibrium_AngI"]   = 7.5
    p["nominal_equilibrium_AngII"]  = 4.75
    p["renin_half_life"]            = 0.1733
    p["AngI_half_life"]             = 0.008333
    p["AngII_half_life"]            = 0.011
    p["AT1_bound_AngII_half_life"]  = 0.2
    p["AT2_bound_AngII_half_life"]  = 0.2
    p["AT1_PRC_slope"]              = -1.2
    p["AT1_PRC_yint"]               = 0.0
    p["fraction_AT1_bound_AngII"]   = 0.75
    p["nominal_ACE_activity"]       = 48.9
    p["nominal_AT1_receptor_binding_rate"]  = 12.1
    p["nominal_AT2_receptor_binding_rate"]  = 4.0
    p["nominal_chymase_activity"]   = 1.25
    p["nominal_equilibrium_AT1_bound_AngII"] = 16.63
    p["nominal_equilibrium_PRC"]    = 16.4

    # ── AT1 vascular / tubular effects ────────────────────────────────────
    p["AT1_svr_slope"]  = 0.0
    p["AT1_preaff_scale"] = 0.5
    p["AT1_preaff_slope"] = 7.0
    p["AT1_aff_scale"]  = 0.5
    p["AT1_aff_slope"]  = 7.0
    p["AT1_eff_scale"]  = 0.3
    p["AT1_eff_slope"]  = 7.0
    p["AT1_PT_scale"]   = 0.1
    p["AT1_PT_slope"]   = 7.0
    p["AT1_aldo_slope"] = 0.05

    # ── Aldosterone ───────────────────────────────────────────────────────
    p["nominal_aldosterone_concentration"] = 85.0
    p["aldo_DCT_scale"]  = 0.0
    p["aldo_DCT_slope"]  = 0.5
    p["aldo_CD_scale"]   = 0.3
    p["aldo_CD_slope"]   = 0.5
    p["aldo_renin_slope"]= -0.05

    # ── Na / water exchange ───────────────────────────────────────────────
    p["Q_water"]         = 1.0
    p["Q_Na"]            = 1.0

    # ── Vasopressin / osmolarity ──────────────────────────────────────────
    p["Na_controller_gain"]   = 2.0
    p["Kp_VP"]                = 0.05
    p["Ki_VP"]                = 0.00002
    p["nom_ADH_urea_permeability"]  = 0.98
    p["nom_ADH_water_permeability"] = 0.98
    p["nominal_vasopressin_conc"]   = 4.0
    p["water_intake_vasopressin_scale"] = 0.0
    p["water_intake_vasopressin_slope"] = -0.5

    # ── TGF ───────────────────────────────────────────────────────────────
    p["S_tubulo_glomerular_feedback"] = 0.7
    p["F_md_scale_tubulo_glomerular_feedback"] = 6.0
    p["MD_Na_concentration_setpoint"] = 62.4

    # ── Macula densa / renin ──────────────────────────────────────────────
    p["md_renin_A"]      = 1.0
    p["md_renin_tau"]    = 2.0

    # ── Arteriole nonlinearity ────────────────────────────────────────────
    p["preaff_signal_nonlin_scale"]   = 3.0
    p["afferent_signal_nonlin_scale"] = 3.0
    p["efferent_signal_nonlin_scale"] = 3.0

    # ── Cardiac output autoregulation ─────────────────────────────────────
    p["tissue_autoreg_scale"] = 1.0
    p["Kp_CO"]               = 1.5
    p["Ki_CO"]               = 30.0

    # ── Myogenic / autoregulation ─────────────────────────────────────────
    p["gp_autoreg_scale"]    = 0.0
    p["preaff_autoreg_scale"]= 0.5
    p["myogenic_steepness"]  = 2.0

    # ── Pressure natriuresis ──────────────────────────────────────────────
    p["pressure_natriuresis_PT_scale"]  = 3.0
    p["pressure_natriuresis_PT_slope"]  = 1.0
    p["pressure_natriuresis_LoH_scale"] = 3.0
    p["pressure_natriuresis_LoH_slope"] = 1.0
    p["pressure_natriuresis_DCT_scale"] = 3.0
    p["pressure_natriuresis_DCT_slope"] = 1.0
    p["pressure_natriuresis_CD_scale"]  = 3.0
    p["pressure_natriuresis_CD_slope"]  = 1.0

    # ── Time constants for ODE delays ─────────────────────────────────────
    p["C_aldo_secretion"]     = 1000.0
    p["C_P_bowmans"]          = 1000.0
    p["C_P_oncotic"]          = 1000.0
    p["C_tgf_reset"]          = 0.0
    p["C_cardiac_output_delayed"] = 0.001
    p["C_co_error"]           = 0.00001
    p["C_vasopressin_delay"]  = 1.0
    p["C_md_flow"]            = 0.001
    p["C_tgf"]                = 1.0
    p["C_na_excretion_na_amount"]  = -1.0
    p["C_na_intake_na_amount"]     = 1.0
    p["C_urine_flow_ecf_volume"]   = -1.0
    p["C_water_intake_ecf_volume"] = 1.0
    p["C_Na_error"]           = 1.0 / 6.0
    p["C_serum_creatinine"]   = 1.0

    # ── Drug effects (no drug at baseline) ───────────────────────────────
    p["HCTZ_effect_on_DT_Na_reabs"]       = 1.0
    p["HCTZ_effect_on_renin_secretion"]   = 1.0
    p["DRI_effect_on_PRA"]                = 1.0
    p["CCB_effect_on_preafferent_resistance"] = 1.0
    p["CCB_effect_on_afferent_resistance"]   = 1.0
    p["CCB_effect_on_efferent_resistance"]   = 1.0
    p["MR_antagonist_effect_on_aldo_MR"]    = 1.0
    p["pct_target_inhibition_ARB"]           = 0.0
    p["pct_target_inhibition_ACEi"]          = 0.0
    p["K_Na_ratio_effect_on_aldo"]           = 1.0

    # ── DKD scenario flags ────────────────────────────────────────────────
    p["scenario"]       = scenario  # "normal" | "dbdb" | "dbdb_unx"
    p["injury_on"]      = True      # enable adaptive/injury ODEs

    # Uninephrectomy: remove one kidney at week 8 (reduces effective nephrons by 50%)
    p["unx_time_min"]   = 8.0 * MIN_PER_WEEK  # min
    p["unx_factor"]     = 0.5   # fraction of nephrons remaining after UNX

    # db/db intake changes (relative to normal)
    p["dbdb_Na_scale"]      = 1.5   # Na intake × 1.5 (text Mahato 2018)
    p["dbdb_water_scale"]   = 5.0   # water intake × 5 (text Mahato 2018)

    # ── Derived nominal quantities ─────────────────────────────────────────
    L_m3 = p["L_m3"]
    visc  = p["viscosity_length_constant"]
    neph  = p["baseline_nephrons"]
    nom_aff_d  = p["nom_afferent_diameter"]
    nom_eff_d  = p["nom_efferent_diameter"]
    nom_MAP    = p["nominal_map_setpoint"]
    nom_RBF    = p["nom_renal_blood_flow_L_min"]
    nom_preaff = p["nom_preafferent_arteriole_resistance"]

    p["nom_preafferent_pressure"] = nom_MAP - nom_RBF * nom_preaff
    p["nom_glomerular_pressure"]  = (
        p["nom_preafferent_pressure"]
        - nom_RBF * (L_m3 * visc / nom_aff_d**4 / neph)
    )
    p["nom_postglomerular_pressure"] = (
        p["nom_preafferent_pressure"]
        - nom_RBF * (L_m3 * visc * (1.0 / nom_aff_d**4 + 1.0 / nom_eff_d**4) / neph)
    )
    p["RIHP0"] = p["nom_postglomerular_pressure"]

    # Nominal GFR
    nom_GFR_mL = (
        p["nom_Kf"]
        * (p["nom_glomerular_pressure"]
           - p["nom_oncotic_pressure_difference"]
           - (p["Pc_pt_mmHg"] + p["P_interstitial_mmHg"]))
        / p["nL_mL"]
        * neph
    )
    nom_GFR_L = nom_GFR_mL / 1000.0

    nom_filtered_Na = nom_GFR_L * p["ref_Na_concentration"]  # mEq/min
    # At nominal BG = 90 mg/dL (normal scenario = BG_nom), SGLT2 reabsorbs additional Na.
    # BG = 90 < RTG = 180 mg/dL, so all filtered glucose is reabsorbed (1:1 Na:glucose).
    # This MUST be included so that nominal_cd_na_reabsorption is calibrated to the
    # actual Na delivery to the CD (not the non-SGLT2 PT delivery alone).
    # NOTE: Using blood_glucose_mg_dl(0, "normal") = 90 ensures BG_excess = 0, which
    # prevents delta_Daa from evolving in the normal scenario (matching the Mahato 2018
    # control condition where C57BL/6J mice are normoglycaemic at BG_nom = 90 mg/dL).
    nom_BG_mmol_L   = blood_glucose_mg_dl(0.0, "normal") / 18.0  # 120/18 = 6.667 mmol/L
    nom_RTG_mmol_L  = p["RTG_normal"] / 18.0                     # 180/18 = 10.0 mmol/L
    nom_SNGFR_L     = nom_GFR_L / neph
    nom_phi_glu_filtered = nom_SNGFR_L * nom_BG_mmol_L           # mmol/min per nephron
    nom_phi_glu_reabs    = min(nom_phi_glu_filtered,
                               nom_RTG_mmol_L * nom_SNGFR_L)     # capped at RTG
    nom_Na_SGLT2_total   = nom_phi_glu_reabs * neph              # total mmol/min (1:1 Na:glu)
    nom_PT_Na_out   = max(0.0,
                          nom_filtered_Na * (1.0 - p["eta_pt_nonSGLT2"]) - nom_Na_SGLT2_total)

    nom_Na_in_AscLoH = nom_PT_Na_out / neph
    p["nom_Na_in_AscLoH"] = nom_Na_in_AscLoH

    p["nom_LoH_Na_outflow"] = nom_PT_Na_out * (1.0 - p["nominal_loh_na_reabsorption"])
    nom_DT_Na_out = p["nom_LoH_Na_outflow"] * (1.0 - p["nominal_dt_na_reabsorption"])
    p["nominal_cd_na_reabsorption"] = max(0.0, min(0.999, 1.0 - p["Na_intake_rate"] / nom_DT_Na_out))

    # Peritubular resistance
    nom_RVR = (nom_MAP - p["P_venous"]) / nom_RBF
    p["nom_peritubular_resistance"] = (
        nom_RVR
        - nom_preaff
        - L_m3 * visc * (1.0 / nom_aff_d**4 + 1.0 / nom_eff_d**4) / neph
    )

    # Systemic arterial resistance
    nom_TPR = nom_MAP / p["CO_nom"]
    p["nom_systemic_arterial_resistance"] = nom_TPR - p["R_venous"]

    # Creatinine synthesis
    p["creatinine_synthesis_rate"] = (
        p["equilibrium_serum_creatinine"] * p["dl_ml"] * nom_GFR_mL
    )

    return p


# ---------------------------------------------------------------------------
# Initial conditions (27 states)
# ---------------------------------------------------------------------------

def initial_conditions(p=None):
    """Return y0 (27 elements) for the M018 mouse model."""
    if p is None:
        p = make_params()

    BV  = p["blood_volume_nom"]
    ECF = p["ECF_nom"]
    Na0 = p["ref_Na_concentration"]

    y0 = [
        8.164,          # 0  AngI
        5.17,           # 1  AngII
        16.6,           # 2  AT1_bound_AngII
        5.5,            # 3  AT2_bound_AngII
        17.845,         # 4  plasma_renin_concentration
        BV,             # 5  blood_volume_L
        ECF,            # 6  extracellular_fluid_volume
        BV * Na0,       # 7  sodium_amount
        ECF * Na0,      # 8  ECF_sodium_amount
        1.0,            # 9  tubulo_glomerular_feedback_effect
        1.0,            # 10 normalized_aldosterone_level_delayed
        1.0,            # 11 preafferent_pressure_autoreg_signal
        1.0,            # 12 glomerular_pressure_autoreg_signal
        p["CO_nom"],    # 13 cardiac_output_delayed
        0.0,            # 14 CO_error
        0.0,            # 15 Na_concentration_error
        1.0,            # 16 normalized_vasopressin_concentration_delayed
        p["nom_LoH_Na_outflow"],               # 17 F0_TGF
        p["Pc_pt_mmHg"] + p["P_interstitial_mmHg"],  # 18 P_bowmans (= lumen + interstitial)
        p["nom_oncotic_pressure_difference"],   # 19 oncotic_pressure_difference
        p["nom_LoH_Na_outflow"] / p["baseline_nephrons"],  # 20 SN_macula_densa_Na_flow_delayed
        p["equilibrium_serum_creatinine"] * BV, # 21 serum_creatinine (mg/dL × L)
        # Note on P_bowmans (state 18): the effective Bowman's capsule pressure that
        # opposes GFR is P_in_pt_mmHg = (P_lumen + P_interstitial) / 133.32, so the
        # correct IC is Pc_pt + P_interstitial, not just Pc_pt alone.  Starting at
        # Pc_pt alone gives SNGFR ≈ 52 nL/min and drives a blood-volume collapse
        # before the solver finds SS.
        # ── M018 slow states ──────────────────────────────────────────────
        0.0,            # 22 delta_Daa       (no dilation at t=0)
        0.0,            # 23 delta_SA        (no hypertrophy)
        0.0,            # 24 delta_Perm      (no glomerulosclerosis)
        0.0,            # 25 delta_Nephrons  (no nephron loss)
        0.0,            # 26 delta_Kalbumin  (baseline sieving)
    ]
    return y0


# ---------------------------------------------------------------------------
# ODE right-hand side (27 states)
# ---------------------------------------------------------------------------

def odes(t, y, p):
    """
    Return dy/dt (27 elements).

    The first 22 states mirror M003 (modified for mouse parameters + SGLT2).
    States 22–26 are the slow DKD extension ODEs.
    """
    # ── unpack all 27 states ────────────────────────────────────────────
    (AngI, AngII, AT1_bound_AngII, AT2_bound_AngII, plasma_renin_concentration,
     blood_volume_L, extracellular_fluid_volume,
     sodium_amount, ECF_sodium_amount,
     tubulo_glomerular_feedback_effect,
     normalized_aldosterone_level_delayed,
     preafferent_pressure_autoreg_signal,
     glomerular_pressure_autoreg_signal,
     cardiac_output_delayed, CO_error, Na_concentration_error,
     normalized_vasopressin_concentration_delayed,
     F0_TGF,
     P_bowmans, oncotic_pressure_difference,
     SN_macula_densa_Na_flow_delayed,
     serum_creatinine,
     delta_Daa, delta_SA, delta_Perm, delta_Nephrons, delta_Kalbumin) = y

    # Clamp slow states to physically valid bounds
    delta_Daa      = max(0.0, min(p["delta_Daa_max"],   delta_Daa))
    delta_SA       = max(0.0, min(p["delta_SA_max"],    delta_SA))
    delta_Perm     = max(0.0, min(p["delta_Perm_max"],  delta_Perm))
    delta_Nephrons = max(0.0, min(0.999,                delta_Nephrons))
    delta_Kalbumin = max(0.0,                           delta_Kalbumin)

# ── frequently-used constants ────────────────────────────────────────
    L_m3       = p["L_m3"]
    visc       = p["viscosity_length_constant"]
    gamma      = p["gamma"]
    pi         = p["pi"]
    mmHg_conv  = p["mmHg_Nperm2_conv"]

    nom_aff_d = p["nom_afferent_diameter"]
    nom_eff_d = p["nom_efferent_diameter"]

    # ── DKD-modified base nephron number ────────────────────────────────
    # UNX halves effective nephrons at unx_time_min (if dbdb_unx scenario)
    neph_base = float(p["baseline_nephrons"])
    if p["scenario"] == "dbdb_unx" and t >= p["unx_time_min"]:
        neph_base = neph_base * p["unx_factor"]
    number_of_functional_nephrons = max(1.0, neph_base * (1.0 - delta_Nephrons))

    # ── DKD-modified Kf (Eq 17 Mahato 2018) ─────────────────────────────
    Kf = max(0.01, p["nom_Kf"] * (1.0 + delta_SA - delta_Perm))

    # ── DKD-modified afferent diameter (via delta_Daa) ───────────────────
    D_aff = nom_aff_d * (1.0 + delta_Daa)

    # ═══════════════════════════════════════════════════════════════════
    # SYSTEMIC HEMODYNAMICS (same structure as M003, mouse-scaled)
    # ═══════════════════════════════════════════════════════════════════

    tissue_autoregulation_signal = max(
        0.1,
        1.0 + p["tissue_autoreg_scale"] * (
            (p["Kp_CO"] / p["CO_scale_species"]) * (cardiac_output_delayed - p["CO_nom"])
            + (p["Ki_CO"] / p["CO_scale_species"]) * CO_error
        )
    )
    AT1_svr_int = 1.0 - p["AT1_svr_slope"] * p["nominal_equilibrium_AT1_bound_AngII"]
    AT1_bound_AngII_effect_on_SVR = AT1_svr_int + p["AT1_svr_slope"] * AT1_bound_AngII
    systemic_arterial_resistance = (
        p["nom_systemic_arterial_resistance"]
        * tissue_autoregulation_signal
        * AT1_bound_AngII_effect_on_SVR
    )

    resistance_to_venous_return = (8.0 * p["R_venous"] + systemic_arterial_resistance) / 31.0
    mean_filling_pressure = (
        p["nom_mean_filling_pressure"]
        + (blood_volume_L / p["BV_scale_species"] - p["blood_volume_nom"])
        / p["venous_compliance"]
    )
    cardiac_output = mean_filling_pressure / resistance_to_venous_return
    total_peripheral_resistance = systemic_arterial_resistance + p["R_venous"]
    mean_arterial_pressure_MAP  = cardiac_output * total_peripheral_resistance

    # ═══════════════════════════════════════════════════════════════════
    # RENAL VASCULATURE
    # ═══════════════════════════════════════════════════════════════════

    AT1_preaff_int = 1.0 - p["AT1_preaff_scale"] / 2.0
    AT1_effect_on_preaff = (
        AT1_preaff_int
        + p["AT1_preaff_scale"]
        * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_preaff_slope"])
    )
    AT1_aff_int = 1.0 - p["AT1_aff_scale"] / 2.0
    AT1_effect_on_aff = (
        AT1_aff_int
        + p["AT1_aff_scale"]
        * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_aff_slope"])
    )
    AT1_eff_int = 1.0 - p["AT1_eff_scale"] / 2.0
    AT1_effect_on_eff = (
        AT1_eff_int
        + p["AT1_eff_scale"]
        * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_eff_slope"])
    )

    # Preafferent resistance
    preaff_sig = (
        AT1_effect_on_preaff
        * preafferent_pressure_autoreg_signal
        * p["CCB_effect_on_preafferent_resistance"]
    )
    preaff_sig_adj = _sigmoid(p["preaff_signal_nonlin_scale"] * (1.0 - preaff_sig)) + 0.5
    preafferent_arteriole_resistance = p["nom_preafferent_arteriole_resistance"] * preaff_sig_adj

    # Afferent resistance: uses DKD-modified D_aff
    nom_afferent_arteriole_resistance = L_m3 * visc / D_aff**4
    aff_sig = (
        tubulo_glomerular_feedback_effect
        * AT1_effect_on_aff
        * glomerular_pressure_autoreg_signal
        * p["CCB_effect_on_afferent_resistance"]
    )
    aff_sig_adj = _sigmoid(p["afferent_signal_nonlin_scale"] * (1.0 - aff_sig)) + 0.5
    afferent_arteriole_resistance = nom_afferent_arteriole_resistance * aff_sig_adj

    # Efferent resistance
    nom_efferent_arteriole_resistance = L_m3 * visc / nom_eff_d**4
    eff_sig = AT1_effect_on_eff * p["CCB_effect_on_efferent_resistance"]
    eff_sig_adj = _sigmoid(p["efferent_signal_nonlin_scale"] * (1.0 - eff_sig)) + 0.5
    efferent_arteriole_resistance = nom_efferent_arteriole_resistance * eff_sig_adj

    peritubular_resistance = p["nom_peritubular_resistance"] * number_of_functional_nephrons

    renal_vascular_resistance = (
        preafferent_arteriole_resistance
        + (afferent_arteriole_resistance
           + efferent_arteriole_resistance
           + peritubular_resistance) / number_of_functional_nephrons
    )
    renal_blood_flow_L_min = (mean_arterial_pressure_MAP - p["P_venous"]) / renal_vascular_resistance
    renal_blood_flow_L_min = max(1e-9, renal_blood_flow_L_min)

    preafferent_pressure = mean_arterial_pressure_MAP - renal_blood_flow_L_min * preafferent_arteriole_resistance
    glomerular_pressure  = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * (preafferent_arteriole_resistance + afferent_arteriole_resistance / number_of_functional_nephrons)
    )
    postglomerular_pressure = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * (preafferent_arteriole_resistance
           + (afferent_arteriole_resistance + efferent_arteriole_resistance) / number_of_functional_nephrons)
    )

    # Autoregulatory signals
    preaff_autoreg_int = 1.0 - p["preaff_autoreg_scale"] / 2.0
    preafferent_pressure_autoreg_function = (
        preaff_autoreg_int
        + p["preaff_autoreg_scale"]
        * _sigmoid(-(preafferent_pressure - p["nom_preafferent_pressure"]) / p["myogenic_steepness"])
    )
    gp_autoreg_int = 1.0 - p["gp_autoreg_scale"] / 2.0
    glomerular_pressure_autoreg_function = (
        gp_autoreg_int
        + p["gp_autoreg_scale"]
        * _sigmoid(-(glomerular_pressure - p["nom_glomerular_pressure"]) / p["myogenic_steepness"])
    )

    # ═══════════════════════════════════════════════════════════════════
    # GLOMERULAR FILTRATION  (DKD-modified Kf)
    # ═══════════════════════════════════════════════════════════════════

    net_filtration_pressure = glomerular_pressure - oncotic_pressure_difference - P_bowmans
    SNGFR_nL_min = Kf * net_filtration_pressure   # nL/min per nephron
    SNGFR_nL_min = max(0.0, SNGFR_nL_min)
    SNGFR_L_min  = SNGFR_nL_min / 1e9            # L/min per nephron
    GFR          = SNGFR_L_min * number_of_functional_nephrons  # L/min
    GFR_ml_min   = GFR * 1000.0

    # Creatinine
    serum_creatinine_concentration = serum_creatinine / blood_volume_L  # mg/dL
    creatinine_clearance_rate = GFR_ml_min * p["dl_ml"] * serum_creatinine_concentration

    # Oncotic pressure
    ppc = p["plasma_protein_concentration"]
    Oncotic_pressure_in = 1.629 * ppc + 0.2935 * ppc**2

    SNRBF_nL_min = 1e9 * renal_blood_flow_L_min / number_of_functional_nephrons
    SNRBF_nL_min = max(SNGFR_nL_min + 1e-30, SNRBF_nL_min)
    plasma_protein_concentration_out = ppc * SNRBF_nL_min / (SNRBF_nL_min - SNGFR_nL_min)
    Oncotic_pressure_out = (
        1.629 * plasma_protein_concentration_out
        + 0.2935 * plasma_protein_concentration_out**2
    )
    oncotic_pressure_avg = (Oncotic_pressure_in + Oncotic_pressure_out) / 2.0

    # ═══════════════════════════════════════════════════════════════════
    # PLASMA SODIUM & VASOPRESSIN
    # ═══════════════════════════════════════════════════════════════════

    Na_concentration     = sodium_amount / blood_volume_L
    ECF_Na_concentration = ECF_sodium_amount / extracellular_fluid_volume

    Na_water_controller = p["Na_controller_gain"] * (
        p["Kp_VP"] * (Na_concentration - p["ref_Na_concentration"])
        + p["Ki_VP"] * Na_concentration_error
    )
    normalized_vasopressin_concentration = 1.0 + Na_water_controller

    water_intake_vasopressin_int = 1.0 - p["water_intake_vasopressin_scale"] / 2.0
    # db/db water intake × 5 — scale nom_water_intake
    water_scale = p["dbdb_water_scale"] if p["scenario"] in ("dbdb", "dbdb_unx") else 1.0
    water_intake = (
        p["water_intake_species_scale"]
        * water_scale
        * (p["nom_water_intake"] / p["min_day"])
        * (water_intake_vasopressin_int
           + p["water_intake_vasopressin_scale"]
           * _sigmoid(
               (normalized_vasopressin_concentration_delayed - 1.0)
               / p["water_intake_vasopressin_slope"]
           ))
    )

    # ═══════════════════════════════════════════════════════════════════
    # TUBULAR FLOW WITH SGLT2 (Mahato 2018 Eqs 7–10)
    # ═══════════════════════════════════════════════════════════════════

    L_pt = p["L_pt_s1_nom"] + p["L_pt_s2_nom"] + p["L_pt_s3_nom"]

    # Filtered Na load per nephron (mmol/min)
    SN_filtered_Na_load = SNGFR_L_min * Na_concentration

    # ── SGLT2 glucose coupling (Mahato 2018 Eqs 3–10) ───────────────────
    # Blood glucose as time-dependent input (mg/dL → mmol/L: ÷18)
    C_glu_mg_dl  = blood_glucose_mg_dl(t, p["scenario"])
    C_glu_mmol_L = C_glu_mg_dl / 18.0   # mmol/L (MW glucose = 180, 10 dL/L)

    # Renal threshold (RTG): elevated in db/db
    RTG = (p["RTG_diabetic"] if p["scenario"] in ("dbdb", "dbdb_unx") else p["RTG_normal"])
    RTG_mmol_L = RTG / 18.0  # mmol/L

    phi_glu_filtered = SNGFR_L_min * C_glu_mmol_L   # mmol/min per nephron
    RC_glucose       = RTG_mmol_L   * SNGFR_L_min    # mmol/min per nephron
    phi_glu_reabs    = min(phi_glu_filtered, RC_glucose)

    # Na via SGLT2: 1:1 molar coupling (simplified from SGLT2 physiology)
    # (SGLT2 is 1 Na+ : 1 glucose; SGLT1 is 2:1; we use simplified 1:1 for the whole PT)
    phi_Na_SGLT2 = phi_glu_reabs  # mmol/min per nephron (Eq 8 Mahato 2018)

    # Pressure-natriuresis and AT1 modulation apply to non-SGLT2 fraction
    PN_arg_PT = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_PT_slope"]
    pn_PT_int = 1.0 - p["pressure_natriuresis_PT_scale"] / 2.0
    pressure_natriuresis_PT_effect = max(
        0.001, pn_PT_int + p["pressure_natriuresis_PT_scale"] * _sigmoid(PN_arg_PT)
    )
    AT1_PT_int = 1.0 - p["AT1_PT_scale"] / 2.0
    AT1_effect_on_PT = (
        AT1_PT_int
        + p["AT1_PT_scale"]
        * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_PT_slope"])
    )
    eta_pt_eff = p["eta_pt_nonSGLT2"] * AT1_effect_on_PT * pressure_natriuresis_PT_effect
    phi_Na_nonSGLT2 = SN_filtered_Na_load * min(1.0, eta_pt_eff)

    # Total PT Na reabsorption (clamped — cannot exceed filtered load)
    phi_Na_PT_total = min(phi_Na_nonSGLT2 + phi_Na_SGLT2, SN_filtered_Na_load * 0.999)
    Na_pt_out = max(SN_filtered_Na_load * 0.001, SN_filtered_Na_load - phi_Na_PT_total)

    # Water leaving PT (follows Na osmotically)
    e_pt_eff = 1.0 - Na_pt_out / max(SN_filtered_Na_load, 1e-30)
    e_pt_eff = max(0.0, min(0.999, e_pt_eff))
    water_out_pt = (SNGFR_L_min / max(SN_filtered_Na_load, 1e-30)) * Na_pt_out
    water_out_pt = max(1e-30, water_out_pt)
    Na_concentration_out_pt = Na_pt_out / water_out_pt

    # ── Loop of Henle ────────────────────────────────────────────────────
    PN_arg_LoH = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_LoH_slope"]
    pn_LoH_int = 1.0 - p["pressure_natriuresis_LoH_scale"] / 2.0
    pressure_natriuresis_LoH_effect = max(
        0.001, pn_LoH_int + p["pressure_natriuresis_LoH_scale"] * _sigmoid(PN_arg_LoH)
    )

    water_in_DescLoH  = water_out_pt
    Na_in_DescLoH     = Na_pt_out
    Na_concentration_in_DescLoH = Na_concentration_out_pt
    Na_out_DescLoH    = Na_in_DescLoH

    nom_Na_in_AscLoH  = p["nom_Na_in_AscLoH"]
    deltaLoH_NaFlow   = p["LoH_flow_dependence"] * (Na_out_DescLoH - nom_Na_in_AscLoH)
    AscLoH_Reab_Rate  = (
        p["nominal_loh_na_reabsorption"] * (nom_Na_in_AscLoH + deltaLoH_NaFlow)
    ) / p["L_lh_des"]
    effective_AscLoH_Reab_Rate = AscLoH_Reab_Rate * pressure_natriuresis_LoH_effect

    exp_arg_desc = min(
        effective_AscLoH_Reab_Rate * p["L_lh_des"],
        Na_in_DescLoH
    ) / max(water_in_DescLoH * Na_concentration_in_DescLoH, 1e-30)
    Na_concentration_out_DescLoH = Na_concentration_in_DescLoH * math.exp(min(exp_arg_desc, 500))
    water_out_DescLoH = water_in_DescLoH * Na_concentration_in_DescLoH / max(Na_concentration_out_DescLoH, 1e-30)
    water_out_DescLoH = max(1e-30, water_out_DescLoH)

    Na_in_AscLoH = Na_out_DescLoH
    Na_concentration_in_AscLoH = Na_concentration_out_DescLoH
    water_in_AscLoH = water_out_DescLoH

    clamped_reab = min(p["L_lh_des"] * effective_AscLoH_Reab_Rate, Na_in_DescLoH)
    denom_exp = max(water_in_DescLoH * Na_concentration_in_DescLoH, 1e-30)
    exp_factor = math.exp(min(clamped_reab / denom_exp, 500))
    Na_concentration_out_AscLoH = (
        Na_concentration_in_AscLoH
        - clamped_reab * exp_factor / max(water_in_DescLoH, 1e-30)
    )
    Na_concentration_out_AscLoH = max(0.0, Na_concentration_out_AscLoH)
    Na_reabsorbed_AscLoH = (Na_concentration_in_AscLoH - Na_concentration_out_AscLoH) * water_in_AscLoH
    Na_out_AscLoH = max(0.0, Na_in_AscLoH - Na_reabsorbed_AscLoH)
    water_out_AscLoH = water_in_AscLoH
    Na_concentration_out_AscLoH = Na_out_AscLoH / max(water_out_AscLoH, 1e-30)

    SN_macula_densa_Na_flow = Na_out_AscLoH
    MD_Na_concentration     = Na_concentration_out_AscLoH

    # TGF signal
    MD_Na_setpoint = p["MD_Na_concentration_setpoint"]
    TGF0 = 1.0 - p["S_tubulo_glomerular_feedback"] / 2.0
    tubulo_glomerular_feedback_signal = (
        TGF0
        + p["S_tubulo_glomerular_feedback"]
        * _sigmoid((MD_Na_setpoint - MD_Na_concentration) / p["F_md_scale_tubulo_glomerular_feedback"])
    )

    # ── Distal Convoluted Tubule ──────────────────────────────────────────
    PN_arg_DCT = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_DCT_slope"]
    pn_DCT_int = 1.0 - p["pressure_natriuresis_DCT_scale"] / 2.0
    pressure_natriuresis_DCT_effect = max(
        0.001, pn_DCT_int + p["pressure_natriuresis_DCT_scale"] * _sigmoid(PN_arg_DCT)
    )

    Aldo_MR = normalized_aldosterone_level_delayed * p["MR_antagonist_effect_on_aldo_MR"]
    aldo_DCT_int = 1.0 - p["aldo_DCT_scale"] / 2.0
    aldo_effect_on_DCT = (
        aldo_DCT_int
        + p["aldo_DCT_scale"]
        * _sigmoid((1.0 - Aldo_MR) / p["aldo_DCT_slope"])
    )
    aldo_CD_int = 1.0 - p["aldo_CD_scale"] / 2.0
    aldo_effect_on_CD = (
        aldo_CD_int
        + p["aldo_CD_scale"]
        * _sigmoid((1.0 - Aldo_MR) / p["aldo_CD_slope"])
    )

    e_dct_sodreab = min(
        1.0,
        p["nominal_dt_na_reabsorption"]
        * aldo_effect_on_DCT
        * pressure_natriuresis_DCT_effect
        * p["HCTZ_effect_on_DT_Na_reabs"]
    )
    PN_arg_CD = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_CD_slope"]
    pn_CD_int = 1.0 - p["pressure_natriuresis_CD_scale"] / 2.0
    pressure_natriuresis_CD_effect = max(
        0.001, pn_CD_int + p["pressure_natriuresis_CD_scale"] * _sigmoid(PN_arg_CD)
    )
    e_cd_sodreab = min(
        1.0,
        p["nominal_cd_na_reabsorption"] * aldo_effect_on_CD * pressure_natriuresis_CD_effect
    )

    water_in_DCT = water_out_AscLoH
    Na_in_DCT    = Na_out_AscLoH
    e_dct_c = min(e_dct_sodreab, 1.0 - 1e-12)
    R_dct = -math.log(1.0 - e_dct_c) / max(p["L_dct"], 1e-10)
    Na_out_DCT = Na_in_DCT * math.exp(-R_dct * p["L_dct"])
    water_out_DCT = water_in_DCT

    # ── Collecting Duct ───────────────────────────────────────────────────
    water_in_CD = water_out_DCT
    Na_in_CD    = Na_out_DCT
    Na_concentration_in_CD = Na_out_DCT / max(water_in_CD, 1e-30)

    e_cd_c = min(e_cd_sodreab, 1.0 - 1e-12)
    R_cd   = -math.log(1.0 - e_cd_c) / max(p["L_cd"], 1e-10)
    Na_out_CD = Na_in_CD * math.exp(-R_cd * p["L_cd"])

    ADH_water_permeability = min(
        1.0, max(0.0, p["nom_ADH_water_permeability"] * normalized_vasopressin_concentration)
    )
    # CD water reabsorption is driven by the medullary interstitial osmolality,
    # which is set by the concentrated fluid at the LoH hairpin tip
    # (= Na_concentration_in_AscLoH = Na_concentration_out_DescLoH ≈ 315 mmol/L
    # at nominal).  Using the dilute ascending-limb EXIT concentration would
    # grossly underestimate the medullary gradient and produce excess urine.
    # This matches the M003 formula exactly (M003 model.py line 795).
    Na_conc_medullary = max(Na_concentration_in_AscLoH, 1.0)  # LoH tip concentration
    max_water_reabs_CD = (
        water_in_CD
        - (Na_concentration_in_CD * water_in_CD - Na_in_CD * (1.0 - math.exp(-R_cd * p["L_cd"])))
        / Na_conc_medullary
    )
    # Guard: CD cannot produce water, and negative max_reabs means no reabsorption.
    max_water_reabs_CD = max(0.0, min(water_in_CD, max_water_reabs_CD))
    water_out_CD = max(0.0, min(water_in_CD, water_in_CD - ADH_water_permeability * max_water_reabs_CD))

    urine_flow_rate         = water_out_CD * number_of_functional_nephrons
    Na_excretion_via_urine  = Na_out_CD   * number_of_functional_nephrons
    # db/db Na intake scaled by 1.5
    Na_scale = p["dbdb_Na_scale"] if p["scenario"] in ("dbdb", "dbdb_unx") else 1.0
    Na_intake = p["Na_intake_rate"] * Na_scale

    # ── Tubular pressures (for P_bowmans state) ──────────────────────────
    B1 = (4.0 * p["tubular_compliance"] + 1.0) * 128.0 * gamma / pi
    tc = p["tubular_compliance"]
    Pc_cd     = p["Pc_cd_mmHg"] * mmHg_conv
    Pc_dt     = p["Pc_dt_mmHg"] * mmHg_conv
    Pc_lh_asc = p["Pc_lh_asc_mmHg"] * mmHg_conv
    Pc_lh_des = p["Pc_lh_des_mmHg"] * mmHg_conv
    Pc_pt_Pa  = p["Pc_pt_mmHg"] * mmHg_conv
    P_interstitial = p["P_interstitial_mmHg"] * mmHg_conv

    mean_cd_water_flow = max(0.0, water_in_CD - water_out_CD) / 2.0
    B2_cd = Pc_cd**(4.0*tc) / max(p["Dc_cd"]**4, 1e-40)
    P_in_cd = (B1 * B2_cd * (mean_cd_water_flow / 1e3) * p["L_cd"])**(1.0/(4.0*tc + 1.0))
    P_in_cd_mmHg = (P_in_cd + P_interstitial) / mmHg_conv

    B2_dt = Pc_dt**(4.0*tc) / max(p["Dc_dt"]**4, 1e-40)
    P_in_dt = (P_in_cd**(4.0*tc + 1.0) + B1 * B2_dt * (max(water_in_DCT, 0) / 1e3) * p["L_dct"])**(1.0/(4.0*tc + 1.0))

    B2_lh_asc = Pc_lh_asc**(4.0*tc) / max(p["Dc_lh"]**4, 1e-40)
    P_in_lh_asc = (P_in_dt**(4.0*tc + 1.0) + B1 * B2_lh_asc * (max(water_in_AscLoH, 0) / 1e3) * p["L_lh_asc"])**(1.0/(4.0*tc + 1.0))

    A_lh_des = effective_AscLoH_Reab_Rate / max(water_in_DescLoH * Na_concentration_in_DescLoH, 1e-30)
    A_lh_des = max(A_lh_des, 1e-30)
    B2_lh_des = (Pc_lh_des**(4.0*tc)) * (max(water_in_DescLoH, 0) / 1e3) / max(p["Dc_lh"]**4 * A_lh_des, 1e-40)
    exp_arg_lhdes = min(A_lh_des * p["L_lh_des"], 500.0)
    _base_lhdes = max(0.0, P_in_lh_asc**(4.0*tc + 1.0) + B1 * B2_lh_des * (1.0 - math.exp(-exp_arg_lhdes)))
    P_in_lh_des = _base_lhdes**(1.0/(4.0*tc + 1.0))

    Na_reabs_per_unit_length = -math.log(max(1.0 - e_pt_eff, 1e-12)) / max(L_pt, 1e-10)
    flow_integral_pt = (SN_filtered_Na_load / max(Na_reabs_per_unit_length, 1e-30)) * (1.0 - math.exp(-Na_reabs_per_unit_length * L_pt))
    B2_pt = Pc_pt_Pa**(4.0*tc) / max(p["Dc_pt_nom"]**4, 1e-40)
    B3_pt = (SNGFR_L_min / 1e3) / max(SN_filtered_Na_load, 1e-30)
    _base_pt = max(0.0, P_in_lh_des**(4.0*tc + 1.0) + B1 * B2_pt * B3_pt * max(0.0, flow_integral_pt))
    P_in_pt = _base_pt**(1.0/(4.0*tc + 1.0))
    P_in_pt_mmHg_formula = (P_in_pt + P_interstitial) / mmHg_conv
    # Physical clamp: Bowman's capsule pressure cannot fall below the terminal
    # tubular pressure (Pc_pt + P_interstitial = 14 + 5 = 19 mmHg for mouse).
    # Without this, the compliance formula (calibrated for human L values) underestimates
    # P_bowmans at mouse SNGFR, creating a spurious high-SNGFR attractor.
    P_in_pt_mmHg = max(p["Pc_pt_mmHg"] + p["P_interstitial_mmHg"], P_in_pt_mmHg_formula)

    # ═══════════════════════════════════════════════════════════════════
    # RAAS
    # ═══════════════════════════════════════════════════════════════════

    AT1_aldo_int = 1.0 - p["AT1_aldo_slope"] * p["nominal_equilibrium_AT1_bound_AngII"]
    AngII_effect_on_aldo = AT1_aldo_int + p["AT1_aldo_slope"] * AT1_bound_AngII
    normalized_aldosterone_level = p["K_Na_ratio_effect_on_aldo"] * AngII_effect_on_aldo

    md_effect_on_renin = p["md_renin_A"] * math.exp(
        -p["md_renin_tau"]
        * (SN_macula_densa_Na_flow_delayed * number_of_functional_nephrons - p["nom_LoH_Na_outflow"])
    )
    _at1_ratio = max(AT1_bound_AngII / p["nominal_equilibrium_AT1_bound_AngII"], 1e-15)
    AT1_PRA_effect = 10.0 ** (
        p["AT1_PRC_slope"] * math.log10(_at1_ratio) + p["AT1_PRC_yint"]
    )
    aldo_renin_int = 1.0 - p["aldo_renin_slope"]
    aldo_effect_on_renin = (
        p["aldo_renin_slope"] * normalized_aldosterone_level_delayed + aldo_renin_int
    )

    plasma_renin_activity = (
        p["concentration_to_renin_activity_conversion_plasma"]
        * plasma_renin_concentration
        * p["DRI_effect_on_PRA"]
    )

    renin_secretion_rate = (
        (math.log(2.0) / p["renin_half_life"])
        * p["nominal_equilibrium_PRC"]
        * AT1_PRA_effect
        * md_effect_on_renin
        * p["HCTZ_effect_on_renin_secretion"]
        * aldo_effect_on_renin
    )

    renin_deg   = math.log(2.0) / p["renin_half_life"]
    AngI_deg    = math.log(2.0) / p["AngI_half_life"]
    AngII_deg   = math.log(2.0) / p["AngII_half_life"]
    AT1_deg     = math.log(2.0) / p["AT1_bound_AngII_half_life"]
    AT2_deg     = math.log(2.0) / p["AT2_bound_AngII_half_life"]

    ACE_act     = p["nominal_ACE_activity"] * (1.0 - p["pct_target_inhibition_ACEi"])
    chymase_act = p["nominal_chymase_activity"]
    AT1_bind    = p["nominal_AT1_receptor_binding_rate"] * (1.0 - p["pct_target_inhibition_ARB"])
    AT2_bind    = p["nominal_AT2_receptor_binding_rate"]

    # ═══════════════════════════════════════════════════════════════════
    # SLOW DKD ODEs (Mahato 2018 Eqs 14–19)
    # ═══════════════════════════════════════════════════════════════════

    # Glomerular pressure damage signal (Eq 14)
    Pgc_nom = p["nom_glomerular_pressure"]
    GP_damage = max(0.0, glomerular_pressure / max(Pgc_nom, 1.0) - 1.0)

    # Afferent arteriole dilation — glucose driven (Eq 19)
    BG_excess = max(0.0, C_glu_mg_dl - p["BG_nom"])   # mg/dL
    if p["injury_on"]:
        d_delta_Daa = (p["delta_Daa_max"] - delta_Daa) * BG_excess / p["tau_daa"]
    else:
        d_delta_Daa = 0.0

    # Glomerular hypertrophy / surface area (Eq 15)
    if p["injury_on"]:
        d_delta_SA = (p["delta_SA_max"] - delta_SA) * GP_damage / p["tau_SA"]
    else:
        d_delta_SA = 0.0

    # Glomerulosclerosis (Eq 16)
    if p["injury_on"]:
        d_delta_Perm = (p["delta_Perm_max"] - delta_Perm) * GP_damage / p["tau_perm"]
    else:
        d_delta_Perm = 0.0

    # Nephron loss (ESTIMATED: same form as glomerulosclerosis, τ = τnephron)
    if p["injury_on"]:
        d_delta_Nephrons = (1.0 - delta_Nephrons) * GP_damage / p["tau_nephron"]
    else:
        d_delta_Nephrons = 0.0

    # Albumin sieving coefficient (ESTIMATED: driven by GP_damage, τ = τalbum)
    if p["injury_on"]:
        d_delta_Kalbumin = GP_damage / p["tau_albumin"]
    else:
        d_delta_Kalbumin = 0.0

    # ═══════════════════════════════════════════════════════════════════
    # ALBUMINURIA (Eqs 11–13 Mahato 2018) — diagnostic output, not ODE
    # ═══════════════════════════════════════════════════════════════════
    # (computed here for reference; actual UAER obtained via compute_outputs)

    # ═══════════════════════════════════════════════════════════════════
    # M003 FAST ODEs (adapted for mouse)
    # ═══════════════════════════════════════════════════════════════════

    dAngI = (
        plasma_renin_activity
        - AngI * (chymase_act + ACE_act)
        - AngI * AngI_deg
    )
    dAngII = (
        AngI * (chymase_act + ACE_act)
        - AngII * AngII_deg
        - AngII * AT1_bind
        - AngII * AT2_bind
    )
    dAT1_bound_AngII = AngII * AT1_bind - AT1_deg * AT1_bound_AngII
    dAT2_bound_AngII = AngII * AT2_bind - AT2_deg * AT2_bound_AngII
    dplasma_renin    = renin_secretion_rate - plasma_renin_concentration * renin_deg

    dblood_volume_L = (
        p["C_water_intake_ecf_volume"] * water_intake
        + p["C_urine_flow_ecf_volume"] * urine_flow_rate
        + p["Q_water"] * (Na_concentration - ECF_Na_concentration)
    )
    dextracellular_fluid_volume = p["Q_water"] * (ECF_Na_concentration - Na_concentration)

    dsodium_amount = (
        p["C_na_excretion_na_amount"] * Na_excretion_via_urine
        + p["C_na_intake_na_amount"] * Na_intake
        + p["Q_Na"] * (ECF_Na_concentration - Na_concentration)
    )
    dECF_sodium_amount = p["Q_Na"] * (Na_concentration - ECF_Na_concentration)

    dtgf_effect = p["C_tgf"] * (tubulo_glomerular_feedback_signal - tubulo_glomerular_feedback_effect)
    dnorm_aldo   = p["C_aldo_secretion"] * (normalized_aldosterone_level - normalized_aldosterone_level_delayed)
    dpreaff_sig  = 500.0 * (preafferent_pressure_autoreg_function - preafferent_pressure_autoreg_signal)
    dgp_sig      = 500.0 * (glomerular_pressure_autoreg_function  - glomerular_pressure_autoreg_signal)
    dCO_delayed  = p["C_cardiac_output_delayed"] * (cardiac_output - cardiac_output_delayed)
    dCO_error    = p["C_co_error"] * (cardiac_output - p["CO_nom"])
    dNa_err      = p["C_Na_error"] * (Na_concentration - p["ref_Na_concentration"])
    dVP_delayed  = p["C_vasopressin_delay"] * (normalized_vasopressin_concentration - normalized_vasopressin_concentration_delayed)
    dF0_TGF      = p["C_tgf_reset"] * (SN_macula_densa_Na_flow * number_of_functional_nephrons - F0_TGF)
    dP_bowmans   = p["C_P_bowmans"] * (P_in_pt_mmHg - P_bowmans)
    doncotic     = p["C_P_oncotic"] * (oncotic_pressure_avg - oncotic_pressure_difference)
    dMD_flow     = p["C_md_flow"] * (SN_macula_densa_Na_flow - SN_macula_densa_Na_flow_delayed)
    dserum_cr    = p["creatinine_synthesis_rate"] - creatinine_clearance_rate

    return [
        dAngI, dAngII, dAT1_bound_AngII, dAT2_bound_AngII, dplasma_renin,
        dblood_volume_L, dextracellular_fluid_volume,
        dsodium_amount, dECF_sodium_amount,
        dtgf_effect, dnorm_aldo,
        dpreaff_sig, dgp_sig,
        dCO_delayed, dCO_error, dNa_err,
        dVP_delayed,
        dF0_TGF,
        dP_bowmans, doncotic,
        dMD_flow,
        dserum_cr,
        d_delta_Daa, d_delta_SA, d_delta_Perm, d_delta_Nephrons, d_delta_Kalbumin,
    ]


# ---------------------------------------------------------------------------
# Derived outputs
# ---------------------------------------------------------------------------

def compute_outputs(y, p, t=0.0):
    """Compute key M018 derived outputs from a state vector."""
    (AngI, AngII, AT1_bound_AngII, AT2_bound_AngII, plasma_renin_concentration,
     blood_volume_L, extracellular_fluid_volume,
     sodium_amount, ECF_sodium_amount,
     tubulo_glomerular_feedback_effect,
     normalized_aldosterone_level_delayed,
     preafferent_pressure_autoreg_signal, glomerular_pressure_autoreg_signal,
     cardiac_output_delayed, CO_error, Na_concentration_error,
     normalized_vasopressin_concentration_delayed, F0_TGF,
     P_bowmans, oncotic_pressure_difference,
     SN_macula_densa_Na_flow_delayed, serum_creatinine,
     delta_Daa, delta_SA, delta_Perm, delta_Nephrons, delta_Kalbumin) = y

    L_m3  = p["L_m3"]
    visc  = p["viscosity_length_constant"]
    neph_base = float(p["baseline_nephrons"])
    if p["scenario"] == "dbdb_unx" and t >= p["unx_time_min"]:
        neph_base = neph_base * p["unx_factor"]
    N_nephrons = max(1.0, neph_base * (1.0 - max(0.0, delta_Nephrons)))
    Kf         = max(0.01, p["nom_Kf"] * (1.0 + max(0.0, delta_SA) - max(0.0, delta_Perm)))
    D_aff      = p["nom_afferent_diameter"] * (1.0 + max(0.0, delta_Daa))
    nom_eff_d  = p["nom_efferent_diameter"]

    AT1_svr_int = 1.0 - p["AT1_svr_slope"] * p["nominal_equilibrium_AT1_bound_AngII"]
    AT1_SVR     = AT1_svr_int + p["AT1_svr_slope"] * AT1_bound_AngII
    tissue_sig  = max(0.1, 1.0 + p["tissue_autoreg_scale"] * (
        p["Kp_CO"] * (cardiac_output_delayed - p["CO_nom"])
        + p["Ki_CO"] * CO_error
    ))
    SAR = p["nom_systemic_arterial_resistance"] * tissue_sig * AT1_SVR
    RtVR = (8.0 * p["R_venous"] + SAR) / 31.0
    mfp  = (p["nom_mean_filling_pressure"]
            + (blood_volume_L - p["blood_volume_nom"]) / p["venous_compliance"])
    CO  = mfp / RtVR
    MAP = CO * (SAR + p["R_venous"])

    AT1_preaff_int = 1.0 - p["AT1_preaff_scale"] / 2.0
    AT1_eff_preaff = (AT1_preaff_int
                      + p["AT1_preaff_scale"]
                      * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_preaff_slope"]))
    AT1_aff_int = 1.0 - p["AT1_aff_scale"] / 2.0
    AT1_eff_aff = (AT1_aff_int
                   + p["AT1_aff_scale"]
                   * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_aff_slope"]))
    AT1_eff_eff_int = 1.0 - p["AT1_eff_scale"] / 2.0
    AT1_eff_eff = (AT1_eff_eff_int
                   + p["AT1_eff_scale"]
                   * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_eff_slope"]))

    preaff_R = p["nom_preafferent_arteriole_resistance"] * (
        _sigmoid(p["preaff_signal_nonlin_scale"] * (1.0 - AT1_eff_preaff * preafferent_pressure_autoreg_signal)) + 0.5)
    aff_R = L_m3 * visc / D_aff**4 * (
        _sigmoid(p["afferent_signal_nonlin_scale"] * (1.0 - tubulo_glomerular_feedback_effect * AT1_eff_aff * glomerular_pressure_autoreg_signal)) + 0.5)
    eff_R = L_m3 * visc / nom_eff_d**4 * (
        _sigmoid(p["efferent_signal_nonlin_scale"] * (1.0 - AT1_eff_eff)) + 0.5)
    peritubu_R = p["nom_peritubular_resistance"] * N_nephrons
    RVR = preaff_R + (aff_R + eff_R + peritubu_R) / N_nephrons
    RBF = (MAP - p["P_venous"]) / RVR

    Pgc = MAP - RBF * (preaff_R + aff_R / N_nephrons)
    NFP = Pgc - oncotic_pressure_difference - P_bowmans
    SNGFR_nL = Kf * max(0.0, NFP)
    SNGFR_L  = SNGFR_nL / 1e9
    GFR_L    = SNGFR_L * N_nephrons
    GFR_mL   = GFR_L * 1000.0

    # Albuminuria (Eqs 11–13 Mahato 2018)
    Kalbumin = p["Kalbumin0"] + max(0.0, delta_Kalbumin)
    phi_alb_filt = Kalbumin * SNGFR_L * p["C_albumin"]              # mg/min per nephron
    phi_alb_reabs = min(phi_alb_filt * p["eta_albumin"], p["RC_albumin"])  # mg/min per nephron
    UAER_per_neph = max(0.0, phi_alb_filt - phi_alb_reabs)
    UAER_mg_min   = UAER_per_neph * N_nephrons
    UAER_ug_day   = UAER_mg_min * 1000.0 * p["min_day"]             # µg/day

    return {
        "MAP_mmHg":                MAP,
        "GFR_mL_min":              GFR_mL,
        "SNGFR_nL_min":            SNGFR_nL,
        "RBF_L_min":               RBF,
        "Pgc_mmHg":                Pgc,
        "Kf_nL_min_mmHg":         Kf,
        "N_nephrons":              N_nephrons,
        "UAER_ug_day":             UAER_ug_day,
        "delta_Daa":               delta_Daa,
        "delta_SA":                delta_SA,
        "delta_Perm":              delta_Perm,
        "delta_Nephrons":          delta_Nephrons,
        "AngII":                   AngII,
        "serum_creatinine_mg_dL":  serum_creatinine / blood_volume_L,
        "blood_glucose_mg_dL":     blood_glucose_mg_dl(t, p["scenario"]),
    }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_to_ss(p=None, t_end=100_000.0, y0=None):
    """Integrate to (approximate) steady state; returns (sol, y_ss)."""
    if p is None:
        p = make_params("normal")
    if y0 is None:
        y0 = initial_conditions(p)

    # Mouse model (M018) uses looser tolerances than human (M003) for performance.
    # The mouse ECF volume (0.006 L) creates stiff Q_water/Q_Na terms.
    # rtol=1e-5/atol=1e-7 gives accurate physiological SS (<0.1% error vs tighter tol).
    sol = solve_ivp(
        fun=lambda t, y: odes(t, y, p),
        t_span=(0.0, t_end),
        y0=y0,
        method="LSODA",
        rtol=1e-5,
        atol=1e-7,
        max_step=100.0,
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed (first pass): {sol.message}")

    # Second pass from SS
    import numpy as _np
    y1 = list(sol.y[:, -1])
    bad = [i for i, v in enumerate(y1) if not _np.isfinite(v)]
    if bad:
        raise RuntimeError(
            f"First SS pass produced non-finite states at indices {bad}: "
            f"{[y1[i] for i in bad]}. "
            "Check model parameters and initial conditions."
        )
    sol2 = solve_ivp(
        fun=lambda t, y: odes(t, y, p),
        t_span=(0.0, t_end),
        y0=y1,
        method="LSODA",
        rtol=1e-5,
        atol=1e-7,
        max_step=100.0,
    )
    if not sol2.success:
        raise RuntimeError(f"Solver failed (second pass): {sol2.message}")

    return sol2, list(sol2.y[:, -1])


def simulate_dkd(scenario="dbdb", t_end_weeks=25.0, p=None, y0_ss=None):
    """
    Run the DKD simulation from steady-state initial conditions.

    scenario : "dbdb"      — db/db diabetic mouse (no nephrectomy)
               "dbdb_unx"  — db/db with uninephrectomy at week 8
    t_end_weeks : simulation duration in weeks
    p    : parameter dict (will be overridden to set scenario)
    y0_ss: steady-state from run_to_ss(normal); if None, computed internally

    Returns (sol, t_weeks_array, outputs_dict)
    """
    if p is None:
        p = make_params(scenario)
    else:
        p = dict(p)
        p["scenario"] = scenario

    # Obtain control SS as initial condition
    if y0_ss is None:
        p_ctrl = make_params("normal")
        _, y_ctrl_ss = run_to_ss(p_ctrl)
        y0_ss = y_ctrl_ss

    t_end_min = t_end_weeks * MIN_PER_WEEK

    sol = solve_ivp(
        fun=lambda t, y: odes(t, y, p),
        t_span=(0.0, t_end_min),
        y0=y0_ss,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
        dense_output=True,
        max_step=MIN_PER_WEEK / 4.0,   # at most 2-day steps
    )
    if not sol.success:
        raise RuntimeError(f"DKD solver failed: {sol.message}")

    # Evaluate at weekly intervals
    t_weeks = np.linspace(0.0, t_end_weeks, int(t_end_weeks) + 1)
    t_min   = t_weeks * MIN_PER_WEEK

    outputs = {k: [] for k in compute_outputs(y0_ss, p, 0.0).keys()}
    for ti_min, ti_weeks in zip(t_min, t_weeks):
        y_i = sol.sol(ti_min)
        o   = compute_outputs(list(y_i), p, ti_min)
        for k in o:
            outputs[k].append(o[k])
    for k in outputs:
        outputs[k] = np.array(outputs[k])
    outputs["t_weeks"] = t_weeks

    return sol, outputs


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    print("=" * 60)
    print("M018 Mahato 2018 — Mouse DKD Model — Self-test")
    print("=" * 60)

    print("\n[1] Mouse baseline (lean control) …")
    p_ctrl = make_params("normal")
    t0 = time.time()
    _, y_ss = run_to_ss(p_ctrl)
    dt = time.time() - t0
    out = compute_outputs(y_ss, p_ctrl)
    print(f"  Elapsed: {dt:.1f}s")
    print(f"  GFR       = {out['GFR_mL_min']:.3f} mL/min  (target 0.30)")
    print(f"  SNGFR     = {out['SNGFR_nL_min']:.1f} nL/min  (target 29)")
    print(f"  Pgc       = {out['Pgc_mmHg']:.1f} mmHg  (target 37.5)")
    print(f"  MAP       = {out['MAP_mmHg']:.1f} mmHg  (target 98)")
    print(f"  N_nephrons= {out['N_nephrons']:.0f}  (target 10 000)")
    print(f"  Kf        = {out['Kf_nL_min_mmHg']:.2f} nL/min/mmHg  (target 3.9)")
    print(f"  UAER      = {out['UAER_ug_day']:.1f} µg/day (ctrl ~30 µg/day)")

    print("\n[2] db/db 25-week DKD simulation …")
    p_dbdb = make_params("dbdb")
    t0 = time.time()
    _, outs = simulate_dkd("dbdb", t_end_weeks=25.0, y0_ss=y_ss)
    dt = time.time() - t0
    print(f"  Elapsed: {dt:.1f}s")
    print(f"  GFR  week 0 → 25: {outs['GFR_mL_min'][0]:.3f} → {outs['GFR_mL_min'][-1]:.3f} mL/min")
    print(f"  Pgc  week 0 → 25: {outs['Pgc_mmHg'][0]:.1f} → {outs['Pgc_mmHg'][-1]:.1f} mmHg")
    print(f"  UAER week 0 → 25: {outs['UAER_ug_day'][0]:.0f} → {outs['UAER_ug_day'][-1]:.0f} µg/day")
    print("\nDone.")
