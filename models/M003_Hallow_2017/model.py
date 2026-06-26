"""
Hallow & Gebremichael (2017) Renal QSP Model — Python port
Original R/RxODE by K. Melissa Hallow, University of Georgia

State variables (22 total, order must match):
  0  AngI
  1  AngII
  2  AT1_bound_AngII
  3  AT2_bound_AngII
  4  plasma_renin_concentration
  5  blood_volume_L
  6  extracellular_fluid_volume
  7  sodium_amount
  8  ECF_sodium_amount
  9  tubulo_glomerular_feedback_effect
 10  normalized_aldosterone_level_delayed
 11  preafferent_pressure_autoreg_signal
 12  glomerular_pressure_autoreg_signal
 13  cardiac_output_delayed
 14  CO_error
 15  Na_concentration_error
 16  normalized_vasopressin_concentration_delayed
 17  F0_TGF  (MD Na concentration setpoint — TGF resetting)
 18  P_bowmans
 19  oncotic_pressure_difference
 20  SN_macula_densa_Na_flow_delayed
 21  serum_creatinine

Solver: scipy.integrate.solve_ivp with LSODA (stiff-friendly).
"""

import math
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable 1/(1+exp(x))."""
    return 1.0 / (1.0 + math.exp(max(-500, min(500, x))))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def make_params():
    """Return a dict of all model parameters (same names as R where possible)."""

    p = {}

    # ---- Unit-conversion constants ----------------------------------------
    p["nL_mL"]               = 1e6
    p["dl_ml"]               = 0.01
    p["L_dL"]                = 10.0
    p["L_mL"]                = 1000.0
    p["L_m3"]                = 0.001
    p["g_mg"]                = 0.001
    p["ng_mg"]               = 1e-6
    p["secs_mins"]           = 60.0
    p["min_hr"]              = 60.0
    p["hr_day"]              = 24.0
    p["min_day"]             = 1440.0
    p["MW_creatinine"]       = 113.12
    p["pi"]                  = 3.14
    p["viscosity_length_constant"] = 1.5e-9
    p["gamma"]               = 1.16667e-5
    p["mmHg_Nperm2_conv"]    = 133.32

    # ---- Species-scaling (human = 1) --------------------------------------
    p["ECF_scale_species"]          = 1.0
    p["BV_scale_species"]           = 1.0
    p["water_intake_species_scale"] = 1.0
    p["CO_scale_species"]           = 1.0

    # ---- Systemic parameters ----------------------------------------------
    p["nominal_map_setpoint"]   = 93.0       # mmHg
    p["CO_nom"]                 = 5.0        # L/min
    p["ECF_nom"]                = 15.0       # L
    p["blood_volume_nom"]       = 5.0        # L
    p["Na_intake_rate"]         = 100.0/24.0/60.0  # mEq/min
    p["nom_water_intake"]       = 2.1        # L/day
    p["ref_Na_concentration"]   = 140.0      # mEq/L
    p["plasma_protein_concentration"] = 7.0  # g/dL
    p["equilibrium_serum_creatinine"] = 0.92 # mg/dL
    p["P_venous"]               = 4.0        # mmHg
    p["R_venous"]               = 3.4        # mmHg
    p["nom_right_atrial_pressure"] = 0.87    # mmHg
    p["nom_mean_filling_pressure"]  = 7.0    # mmHg
    p["venous_compliance"]      = 0.13

    # ---- Renal parameters -------------------------------------------------
    p["nom_renal_blood_flow_L_min"] = 1.0   # L/min
    p["baseline_nephrons"]      = 2e6
    p["nom_Kf"]                 = 3.9       # nL/min/mmHg
    p["nom_oncotic_pressure_difference"] = 28.0  # mmHg
    p["P_renal_vein"]           = 4.0

    # Renal vasculature
    p["nom_preafferent_arteriole_resistance"] = 19.0  # mmHg/(L/min)
    p["nom_afferent_diameter"]  = 1.5e-5    # m
    p["nom_efferent_diameter"]  = 1.1e-5    # m

    # Renal tubule geometry
    p["Dc_pt_nom"]   = 27e-6    # m
    p["Dc_lh"]       = 17e-6    # m
    p["Dc_dt"]       = 17e-6    # m
    p["Dc_cd"]       = 22e-6    # m
    p["L_pt_s1_nom"] = 0.005    # m
    p["L_pt_s2_nom"] = 0.005    # m
    p["L_pt_s3_nom"] = 0.004    # m
    p["L_lh_des"]    = 0.01     # m
    p["L_lh_asc"]    = 0.01     # m
    p["L_dct"]       = 0.005    # m
    p["L_cd"]        = 0.01     # m  (= L_lh_des)
    p["tubular_compliance"] = 0.2

    # Tubular pressures (mmHg)
    p["Pc_pt_mmHg"]     = 14.0
    p["Pc_lh_des_mmHg"] = 10.5
    p["Pc_lh_asc_mmHg"] = 7.0
    p["Pc_dt_mmHg"]     = 3.0
    p["Pc_cd_mmHg"]     = 2.0
    p["P_interstitial_mmHg"] = 5.0

    # Na reabsorption fractions
    p["nominal_pt_na_reabsorption"]  = 0.70
    p["nominal_loh_na_reabsorption"] = 0.80
    p["nominal_dt_na_reabsorption"]  = 0.50
    p["LoH_flow_dependence"]         = 1.0

    # ---- RAAS nominal values (literature) ---------------------------------
    p["concentration_to_renin_activity_conversion_plasma"] = 61.0
    p["nominal_equilibrium_PRA"]    = 1000.0   # fmol/mL/hr
    p["nominal_equilibrium_AngI"]   = 7.5      # fmol/mL
    p["nominal_equilibrium_AngII"]  = 4.75     # fmol/mL
    p["nominal_renin_half_life"]    = 0.1733   # hr
    p["nominal_AngI_half_life"]     = 0.5/60.0 # hr
    p["nominal_AngII_half_life"]    = 0.66/60.0 # hr
    p["nominal_AT1_bound_AngII_half_life"] = 12.0/60.0  # hr
    p["nominal_AT2_bound_AngII_half_life"] = 12.0/60.0  # hr
    p["ACE_chymase_fraction"]       = 0.95
    p["fraction_AT1_bound_AngII"]   = 0.75

    # ---- RAAS parameters (overriding the computed values) ----------------
    p["AngI_half_life"]             = 0.008333  # hr
    p["AngII_half_life"]            = 0.011     # hr
    p["AT1_bound_AngII_half_life"]  = 0.2       # hr
    p["AT2_bound_AngII_half_life"]  = 0.2       # hr
    p["AT1_PRC_slope"]              = -1.2
    p["AT1_PRC_yint"]               = 0.0
    p["concentration_to_renin_activity_conversion_plasma"] = 61.0
    p["fraction_AT1_bound_AngII"]   = 0.75
    p["nominal_ACE_activity"]       = 48.9
    p["nominal_AT1_receptor_binding_rate"]  = 12.1
    p["nominal_AT2_receptor_binding_rate"]  = 4.0
    p["nominal_chymase_activity"]   = 1.25
    p["nominal_equilibrium_AT1_bound_AngII"] = 16.63
    p["nominal_equilibrium_PRC"]    = 16.4
    p["renin_half_life"]            = 0.1733

    # ---- AT1 effects -------------------------------------------------------
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

    # ---- Aldosterone -------------------------------------------------------
    p["nominal_aldosterone_concentration"] = 85.0
    p["aldo_DCT_scale"]  = 0.0
    p["aldo_DCT_slope"]  = 0.5
    p["aldo_CD_scale"]   = 0.3
    p["aldo_CD_slope"]   = 0.5
    p["aldo_renin_slope"]= -0.05

    # ---- Na / water exchange -----------------------------------------------
    p["Q_water"]         = 1.0
    p["Q_Na"]            = 1.0

    # ---- Vasopressin / osmolarity ------------------------------------------
    p["Na_controller_gain"]   = 2.0
    p["Kp_VP"]                = 0.05
    p["Ki_VP"]                = 0.00002
    p["nom_ADH_urea_permeability"]  = 0.98
    p["nom_ADH_water_permeability"] = 0.98
    p["nominal_vasopressin_conc"]   = 4.0
    p["water_intake_vasopressin_scale"] = 0.0
    p["water_intake_vasopressin_slope"] = -0.5

    # ---- TGF ---------------------------------------------------------------
    p["S_tubulo_glomerular_feedback"] = 0.7
    p["F_md_scale_tubulo_glomerular_feedback"] = 6.0
    p["MD_Na_concentration_setpoint"] = 62.4

    # ---- Macula densa / renin ----------------------------------------------
    p["md_renin_A"]      = 1.0
    p["md_renin_tau"]    = 2.0

    # ---- Arteriole nonlinearity --------------------------------------------
    p["preaff_signal_nonlin_scale"]   = 3.0
    p["afferent_signal_nonlin_scale"] = 3.0
    p["efferent_signal_nonlin_scale"] = 3.0

    # ---- Tissue autoregulation (cardiac output) ----------------------------
    p["tissue_autoreg_scale"] = 1.0
    p["Kp_CO"]               = 1.5
    p["Ki_CO"]               = 30.0

    # ---- Myogenic / autoregulation -----------------------------------------
    p["gp_autoreg_scale"]    = 0.0
    p["preaff_autoreg_scale"]= 0.5
    p["myogenic_steepness"]  = 2.0

    # ---- Pressure natriuresis ----------------------------------------------
    p["pressure_natriuresis_PT_scale"]  = 3.0
    p["pressure_natriuresis_PT_slope"]  = 1.0
    p["pressure_natriuresis_LoH_scale"] = 3.0
    p["pressure_natriuresis_LoH_slope"] = 1.0
    p["pressure_natriuresis_DCT_scale"] = 3.0
    p["pressure_natriuresis_DCT_slope"] = 1.0
    p["pressure_natriuresis_CD_scale"]  = 3.0
    p["pressure_natriuresis_CD_slope"]  = 1.0

    # ---- Time constants for ODE delays ------------------------------------
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
    p["C_Na_error"]           = 1.0/6.0
    p["C_serum_creatinine"]   = 1.0

    # ---- Drug effects (baseline = no drug) --------------------------------
    p["HCTZ_effect_on_DT_Na_reabs"]       = 1.0
    p["HCTZ_effect_on_renin_secretion"]   = 1.0
    p["DRI_effect_on_PRA"]                = 1.0
    p["CCB_effect_on_preafferent_resistance"] = 1.0
    p["CCB_effect_on_afferent_resistance"]   = 1.0
    p["CCB_effect_on_efferent_resistance"]   = 1.0
    p["MR_antagonist_effect_on_aldo_MR"]    = 1.0
    p["pct_target_inhibition_ARB"]           = 0.0
    p["pct_target_inhibition_ACEi"]          = 0.0

    # ---- K/Na ratio effect on aldo -----------------------------------------
    p["K_Na_ratio_effect_on_aldo"] = 1.0

    # ---- Derived nominal quantities (computed exactly as in calcNomParams) --
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
        - nom_RBF * (L_m3 * visc * (1.0/nom_aff_d**4 + 1.0/nom_eff_d**4) / neph)
    )
    p["RIHP0"] = p["nom_postglomerular_pressure"]

    # Nominal GFR (mL/min)
    nom_GFR_mL = (
        p["nom_Kf"]
        * (p["nom_glomerular_pressure"]
           - p["nom_oncotic_pressure_difference"]
           - (p["Pc_pt_mmHg"] + p["P_interstitial_mmHg"]))
        / p["nL_mL"]
        * neph
    )
    # (in L/min for internal use)
    nom_GFR_L = nom_GFR_mL / 1000.0

    nom_filtered_Na = nom_GFR_L * p["ref_Na_concentration"]  # mEq/min
    nom_PT_Na_out   = nom_filtered_Na * (1.0 - p["nominal_pt_na_reabsorption"])

    # SN Na into ascending LoH per nephron (mEq/min per nephron)
    nom_Na_in_AscLoH = nom_PT_Na_out / neph
    p["nom_Na_in_AscLoH"] = nom_Na_in_AscLoH

    p["nom_LoH_Na_outflow"] = nom_PT_Na_out * (1.0 - p["nominal_loh_na_reabsorption"])
    nom_DT_Na_out = p["nom_LoH_Na_outflow"] * (1.0 - p["nominal_dt_na_reabsorption"])
    p["nominal_cd_na_reabsorption"] = 1.0 - p["Na_intake_rate"] / nom_DT_Na_out

    # Peritubular resistance
    nom_RVR = (nom_MAP - p["P_venous"]) / nom_RBF
    p["nom_peritubular_resistance"] = (
        nom_RVR
        - nom_preaff
        - L_m3 * visc * (1.0/nom_aff_d**4 + 1.0/nom_eff_d**4) / neph
    )

    # Systemic resistance
    nom_TPR = nom_MAP / p["CO_nom"]
    p["nom_systemic_arterial_resistance"] = nom_TPR - p["R_venous"]

    # Creatinine synthesis rate
    p["creatinine_synthesis_rate"] = (
        p["equilibrium_serum_creatinine"] * p["dl_ml"] * nom_GFR_mL
    )

    return p


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def initial_conditions(p=None):
    """Return y0 list (22 states) matching the R runModel.R inits."""
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
        p["nom_LoH_Na_outflow"],  # 17 F0_TGF (total LoH Na outflow)
        p["Pc_pt_mmHg"],          # 18 P_bowmans
        p["nom_oncotic_pressure_difference"],  # 19 oncotic_pressure_difference
        p["nom_LoH_Na_outflow"] / p["baseline_nephrons"],  # 20 SN_macula_densa_Na_flow_delayed
        p["equilibrium_serum_creatinine"] * p["blood_volume_nom"],  # 21 serum_creatinine  (mg/dL * L)
    ]
    return y0


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def odes(t, y, p):
    """
    Return dy/dt as a Python list — scipy.integrate.solve_ivp convention.
    All variables follow the R modelfile.R equations exactly.
    """
    # --- unpack state variables ---
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
     serum_creatinine) = y

    # --- unpack frequently-used parameters ---
    L_m3  = p["L_m3"]
    visc  = p["viscosity_length_constant"]
    neph  = p["baseline_nephrons"]
    gamma = p["gamma"]
    pi    = p["pi"]
    mmHg_conv = p["mmHg_Nperm2_conv"]

    nom_aff_d = p["nom_afferent_diameter"]
    nom_eff_d = p["nom_efferent_diameter"]

    # =========================================================
    # SYSTEMIC HEMODYNAMICS
    # =========================================================

    # Tissue autoregulation signal (PI controller of TPR)
    tissue_autoregulation_signal = max(
        0.1,
        1.0 + p["tissue_autoreg_scale"] * (
            (p["Kp_CO"] / p["CO_scale_species"]) * (cardiac_output_delayed - p["CO_nom"])
            + (p["Ki_CO"] / p["CO_scale_species"]) * CO_error
        )
    )

    # AT1-bound AngII effect on SVR
    AT1_svr_int = 1.0 - p["AT1_svr_slope"] * p["nominal_equilibrium_AT1_bound_AngII"]
    AT1_bound_AngII_effect_on_SVR = AT1_svr_int + p["AT1_svr_slope"] * AT1_bound_AngII

    systemic_arterial_resistance = (
        p["nom_systemic_arterial_resistance"]
        * tissue_autoregulation_signal
        * AT1_bound_AngII_effect_on_SVR
    )

    # Cardiac output
    resistance_to_venous_return = (8.0 * p["R_venous"] + systemic_arterial_resistance) / 31.0
    mean_filling_pressure = (
        p["nom_mean_filling_pressure"]
        + (blood_volume_L / p["BV_scale_species"] - p["blood_volume_nom"])
        / p["venous_compliance"]
    )
    cardiac_output = mean_filling_pressure / resistance_to_venous_return

    # MAP
    total_peripheral_resistance = systemic_arterial_resistance + p["R_venous"]
    mean_arterial_pressure_MAP  = cardiac_output * total_peripheral_resistance

    # =========================================================
    # RENAL VASCULATURE
    # =========================================================

    # AT1-bound AngII effects on arterioles
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

    # Preafferent arteriole resistance
    preaff_arteriole_signal_multiplier = (
        AT1_effect_on_preaff
        * preafferent_pressure_autoreg_signal
        * p["CCB_effect_on_preafferent_resistance"]
    )
    preaff_arteriole_adjusted_signal_multiplier = (
        _sigmoid(p["preaff_signal_nonlin_scale"] * (1.0 - preaff_arteriole_signal_multiplier))
        + 0.5
    )
    preafferent_arteriole_resistance = (
        p["nom_preafferent_arteriole_resistance"]
        * preaff_arteriole_adjusted_signal_multiplier
    )

    # Nominal afferent and efferent resistances
    nom_afferent_arteriole_resistance = L_m3 * visc / nom_aff_d**4
    nom_efferent_arteriole_resistance = L_m3 * visc / nom_eff_d**4

    # Afferent arteriole resistance
    afferent_arteriole_signal_multiplier = (
        tubulo_glomerular_feedback_effect
        * AT1_effect_on_aff
        * glomerular_pressure_autoreg_signal
        * p["CCB_effect_on_afferent_resistance"]
    )
    afferent_arteriole_adjusted_signal_multiplier = (
        _sigmoid(p["afferent_signal_nonlin_scale"] * (1.0 - afferent_arteriole_signal_multiplier))
        + 0.5
    )
    afferent_arteriole_resistance = (
        nom_afferent_arteriole_resistance
        * afferent_arteriole_adjusted_signal_multiplier
    )

    # Efferent arteriole resistance
    efferent_arteriole_signal_multiplier = (
        AT1_effect_on_eff * p["CCB_effect_on_efferent_resistance"]
    )
    efferent_arteriole_adjusted_signal_multiplier = (
        _sigmoid(p["efferent_signal_nonlin_scale"] * (1.0 - efferent_arteriole_signal_multiplier))
        + 0.5
    )
    efferent_arteriole_resistance = (
        nom_efferent_arteriole_resistance
        * efferent_arteriole_adjusted_signal_multiplier
    )

    # Peritubular resistance
    peritubular_resistance = p["nom_peritubular_resistance"] * neph

    # Renal vascular resistance
    renal_vascular_resistance = (
        preafferent_arteriole_resistance
        + (afferent_arteriole_resistance
           + efferent_arteriole_resistance
           + peritubular_resistance) / neph
    )

    # Renal blood flow
    renal_blood_flow_L_min = (mean_arterial_pressure_MAP - p["P_venous"]) / renal_vascular_resistance

    # Renal pressures
    preafferent_pressure = mean_arterial_pressure_MAP - renal_blood_flow_L_min * preafferent_arteriole_resistance
    glomerular_pressure  = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * (preafferent_arteriole_resistance + afferent_arteriole_resistance / neph)
    )
    postglomerular_pressure = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * (preafferent_arteriole_resistance
           + (afferent_arteriole_resistance + efferent_arteriole_resistance) / neph)
    )

    # Autoregulatory signals (states that track these with fast time constant)
    # R formula: int + scale / (1+exp((nom_P - P) / myogenic_steepness))
    # = int + scale * _sigmoid(-(P - nom_P)/steep)  since _sigmoid(x)=1/(1+exp(x))
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

    # =========================================================
    # GLOMERULAR FILTRATION
    # =========================================================

    number_of_functional_nephrons = neph

    net_filtration_pressure = glomerular_pressure - oncotic_pressure_difference - P_bowmans
    SNGFR_nL_min = p["nom_Kf"] * net_filtration_pressure  # nL/min per nephron
    GFR = SNGFR_nL_min / 1e6 / 1000.0 * number_of_functional_nephrons  # L/min
    GFR_ml_min = GFR * 1000.0

    # Serum creatinine
    # serum_creatinine state = eq_cr[mg/dL]*BV[L], so state/BV[L] = concentration in mg/dL
    serum_creatinine_concentration = serum_creatinine / blood_volume_L  # mg/dL
    creatinine_clearance_rate = GFR_ml_min * p["dl_ml"] * serum_creatinine_concentration

    # Oncotic pressure
    ppc = p["plasma_protein_concentration"]
    Oncotic_pressure_in = 1.629 * ppc + 0.2935 * ppc**2

    SNRBF_nl_min = 1e6 * 1000.0 * renal_blood_flow_L_min / number_of_functional_nephrons
    plasma_protein_concentration_out = (
        SNRBF_nl_min * ppc / (SNRBF_nl_min - SNGFR_nL_min)
    )
    Oncotic_pressure_out = (
        1.629 * plasma_protein_concentration_out
        + 0.2935 * plasma_protein_concentration_out**2
    )
    oncotic_pressure_avg = (Oncotic_pressure_in + Oncotic_pressure_out) / 2.0

    # =========================================================
    # PLASMA SODIUM & VASOPRESSIN
    # =========================================================

    Na_concentration     = sodium_amount / blood_volume_L       # mEq/L
    ECF_Na_concentration = ECF_sodium_amount / extracellular_fluid_volume

    Na_water_controller = p["Na_controller_gain"] * (
        p["Kp_VP"] * (Na_concentration - p["ref_Na_concentration"])
        + p["Ki_VP"] * Na_concentration_error
    )
    normalized_vasopressin_concentration = 1.0 + Na_water_controller
    # vasopressin_concentration = p["nominal_vasopressin_conc"] * normalized_vasopressin_concentration

    water_intake_vasopressin_int = 1.0 - p["water_intake_vasopressin_scale"] / 2.0
    water_intake = (
        p["water_intake_species_scale"]
        * (p["nom_water_intake"] / 60.0 / 24.0)
        * (water_intake_vasopressin_int
           + p["water_intake_vasopressin_scale"]
           * _sigmoid(
               (normalized_vasopressin_concentration_delayed - 1.0)
               / p["water_intake_vasopressin_slope"]
           ))
    )

    # =========================================================
    # TUBULAR FLOW AND REABSORPTION
    # =========================================================

    Dc_pt = p["Dc_pt_nom"]
    L_pt  = p["L_pt_s1_nom"] + p["L_pt_s2_nom"] + p["L_pt_s3_nom"]

    # Filtered Na load (per nephron, mEq/min)
    SN_filtered_Na_load = (SNGFR_nL_min / 1e6 / 1000.0) * Na_concentration
    filtered_Na_load    = SN_filtered_Na_load * number_of_functional_nephrons

    # --- Pressure natriuresis effects ---
    PN_arg_PT  = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_PT_slope"]
    PN_arg_LoH = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_LoH_slope"]
    PN_arg_DCT = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_DCT_slope"]
    PN_arg_CD  = (postglomerular_pressure - p["RIHP0"]) / p["pressure_natriuresis_CD_slope"]

    # R formula: int + scale / (1+exp(+(postglom_P - RIHP0)/slope))
    # = int + scale * _sigmoid(+PN_arg)   [_sigmoid(x) = 1/(1+exp(x))]
    pn_PT_int = 1.0 - p["pressure_natriuresis_PT_scale"] / 2.0
    pressure_natriuresis_PT_effect = max(
        0.001,
        pn_PT_int + p["pressure_natriuresis_PT_scale"] * _sigmoid(PN_arg_PT)
    )

    pn_LoH_int = 1.0 - p["pressure_natriuresis_LoH_scale"] / 2.0
    pressure_natriuresis_LoH_effect = max(
        0.001,
        pn_LoH_int + p["pressure_natriuresis_LoH_scale"] * _sigmoid(PN_arg_LoH)
    )

    pn_DCT_magnitude = max(0.0, p["pressure_natriuresis_DCT_scale"])
    pn_DCT_int = 1.0 - pn_DCT_magnitude / 2.0
    pressure_natriuresis_DCT_effect = max(
        0.001,
        pn_DCT_int + pn_DCT_magnitude * _sigmoid(PN_arg_DCT)
    )

    pn_CD_magnitude = max(0.0, p["pressure_natriuresis_CD_scale"])
    pn_CD_int = 1.0 - pn_CD_magnitude / 2.0
    pressure_natriuresis_CD_effect = max(
        0.001,
        pn_CD_int + pn_CD_magnitude * _sigmoid(PN_arg_CD)
    )

    # AT1-bound AngII effect on PT reabsorption
    AT1_PT_int = 1.0 - p["AT1_PT_scale"] / 2.0
    AT1_effect_on_PT = (
        AT1_PT_int
        + p["AT1_PT_scale"]
        * _sigmoid(-(AT1_bound_AngII - p["nominal_equilibrium_AT1_bound_AngII"]) / p["AT1_PT_slope"])
    )

    # Aldosterone effects
    Aldo_MR_normalised_effect = normalized_aldosterone_level_delayed * p["MR_antagonist_effect_on_aldo_MR"]

    # R: int + scale / (1+exp((1-Aldo)/slope)) = int + scale * _sigmoid((1-Aldo)/slope)
    aldo_DCT_int = 1.0 - p["aldo_DCT_scale"] / 2.0
    aldo_effect_on_DCT = (
        aldo_DCT_int
        + p["aldo_DCT_scale"]
        * _sigmoid((1.0 - Aldo_MR_normalised_effect) / p["aldo_DCT_slope"])
    )

    aldo_CD_int = 1.0 - p["aldo_CD_scale"] / 2.0
    aldo_effect_on_CD = (
        aldo_CD_int
        + p["aldo_CD_scale"]
        * _sigmoid((1.0 - Aldo_MR_normalised_effect) / p["aldo_CD_slope"])
    )

    # Fractional reabsorption rates
    e_pt_sodreab = min(
        1.0,
        p["nominal_pt_na_reabsorption"] * AT1_effect_on_PT * pressure_natriuresis_PT_effect
    )
    e_dct_sodreab = min(
        1.0,
        p["nominal_dt_na_reabsorption"]
        * aldo_effect_on_DCT
        * pressure_natriuresis_DCT_effect
        * p["HCTZ_effect_on_DT_Na_reabs"]
    )
    e_cd_sodreab = min(
        1.0,
        p["nominal_cd_na_reabsorption"] * aldo_effect_on_CD * pressure_natriuresis_CD_effect
    )

    # --- Proximal Tubule ---
    e_pt_clamped = min(e_pt_sodreab, 1.0 - 1e-12)
    Na_reabs_per_unit_length = -math.log(1.0 - e_pt_clamped) / L_pt
    Na_pt_out = SN_filtered_Na_load * math.exp(-Na_reabs_per_unit_length * L_pt)
    water_out_pt = ((SNGFR_nL_min / 1e6 / 1000.0) / SN_filtered_Na_load) * Na_pt_out
    Na_concentration_out_pt = Na_pt_out / water_out_pt

    # --- Loop of Henle ---
    water_in_DescLoH = water_out_pt
    Na_in_DescLoH    = Na_pt_out
    Na_concentration_in_DescLoH = Na_concentration_out_pt

    # No solute reabsorption in descending limb
    Na_out_DescLoH = Na_in_DescLoH

    # Ascending limb reabsorption rate
    nom_Na_in_AscLoH = p["nom_Na_in_AscLoH"]
    deltaLoH_NaFlow  = p["LoH_flow_dependence"] * (Na_out_DescLoH - nom_Na_in_AscLoH)
    AscLoH_Reab_Rate = (
        p["nominal_loh_na_reabsorption"]
        * (nom_Na_in_AscLoH + deltaLoH_NaFlow)
    ) / p["L_lh_des"]
    effective_AscLoH_Reab_Rate = AscLoH_Reab_Rate * pressure_natriuresis_LoH_effect

    # Descending limb water reabsorption
    exp_arg_desc = min(
        effective_AscLoH_Reab_Rate * p["L_lh_des"],
        Na_in_DescLoH
    ) / (water_in_DescLoH * Na_concentration_in_DescLoH)
    Na_concentration_out_DescLoH = Na_concentration_in_DescLoH * math.exp(exp_arg_desc)
    water_out_DescLoH = water_in_DescLoH * Na_concentration_in_DescLoH / Na_concentration_out_DescLoH

    # Ascending limb
    Na_in_AscLoH  = Na_out_DescLoH
    Na_concentration_in_AscLoH = Na_concentration_out_DescLoH
    water_in_AscLoH = water_out_DescLoH

    clamped_reab = min(p["L_lh_des"] * effective_AscLoH_Reab_Rate, Na_in_DescLoH)
    exp_factor   = math.exp(clamped_reab / (water_in_DescLoH * Na_concentration_in_DescLoH))
    Na_concentration_out_AscLoH = (
        Na_concentration_in_AscLoH
        - clamped_reab * exp_factor / water_in_DescLoH
    )
    Na_reabsorbed_AscLoH = (Na_concentration_in_AscLoH - Na_concentration_out_AscLoH) * water_in_AscLoH
    Na_out_AscLoH = max(0.0, Na_in_AscLoH - Na_reabsorbed_AscLoH)

    water_out_AscLoH = water_in_AscLoH  # impermeable to water
    Na_concentration_out_AscLoH = Na_out_AscLoH / water_out_AscLoH

    # Macula densa
    SN_macula_densa_Na_flow = Na_out_AscLoH
    MD_Na_concentration     = Na_concentration_out_AscLoH

    # TGF — R formula: TGF0 + S/(1+exp((setpoint - MD_Na)/F_scale))
    # = TGF0 + S * _sigmoid((setpoint - MD_Na)/F_scale)
    MD_Na_concentration_setpoint = p["MD_Na_concentration_setpoint"]
    TGF0 = 1.0 - p["S_tubulo_glomerular_feedback"] / 2.0
    tubulo_glomerular_feedback_signal = (
        TGF0
        + p["S_tubulo_glomerular_feedback"]
        * _sigmoid(
            (MD_Na_concentration_setpoint - MD_Na_concentration)
            / p["F_md_scale_tubulo_glomerular_feedback"]
        )
    )

    # --- Distal Convoluted Tubule ---
    water_in_DCT = water_out_AscLoH
    Na_in_DCT    = Na_out_AscLoH
    water_out_DCT = water_in_DCT  # impermeable to water

    # Clamp to avoid log(0) when reabsorption fraction saturates to 1
    e_dct_clamped = min(e_dct_sodreab, 1.0 - 1e-12)
    R_dct = -math.log(1.0 - e_dct_clamped) / p["L_dct"]
    Na_out_DCT = Na_in_DCT * math.exp(-R_dct * p["L_dct"])
    water_out_DCT_c = water_out_DCT

    # --- Collecting Duct ---
    water_in_CD = water_out_DCT_c
    Na_in_CD    = Na_out_DCT
    Na_concentration_in_CD = Na_out_DCT / water_in_CD

    e_cd_clamped = min(e_cd_sodreab, 1.0 - 1e-12)
    R_cd    = -math.log(1.0 - e_cd_clamped) / p["L_cd"]
    Na_out_CD = Na_in_CD * math.exp(-R_cd * p["L_cd"])

    ADH_water_permeability = min(
        1.0,
        max(0.0, p["nom_ADH_water_permeability"] * normalized_vasopressin_concentration)
    )
    max_water_reabs_CD = (
        water_in_CD
        - (Na_concentration_in_CD * water_in_CD - Na_in_CD * (1.0 - math.exp(-R_cd * p["L_cd"])))
        / Na_concentration_in_AscLoH
    )
    water_out_CD = max(0.0, water_in_CD - ADH_water_permeability * max_water_reabs_CD)

    # Urine
    urine_flow_rate       = water_out_CD * number_of_functional_nephrons
    daily_urine_flow      = urine_flow_rate * 60.0 * 24.0
    Na_excretion_via_urine = Na_out_CD * number_of_functional_nephrons
    daily_water_intake     = water_intake * 24.0 * 60.0

    # =========================================================
    # TUBULAR PRESSURES (not needed for ODEs but included for completeness)
    # =========================================================
    # (These fast-ODE states P_bowmans and oncotic_pressure_difference are
    #  driven by P_in_pt_mmHg and oncotic_pressure_avg computed below)

    B1 = (4.0 * p["tubular_compliance"] + 1.0) * 128.0 * gamma / pi
    Pc_cd = p["Pc_cd_mmHg"] * mmHg_conv
    Pc_dt = p["Pc_dt_mmHg"] * mmHg_conv
    Pc_lh_asc = p["Pc_lh_asc_mmHg"] * mmHg_conv
    Pc_lh_des = p["Pc_lh_des_mmHg"] * mmHg_conv
    Pc_pt_Pa  = p["Pc_pt_mmHg"] * mmHg_conv
    P_interstitial = p["P_interstitial_mmHg"] * mmHg_conv
    tc = p["tubular_compliance"]

    mean_cd_water_flow = (water_in_CD - water_out_CD) / 2.0
    B2_cd = Pc_cd**(4.0*tc) / p["Dc_cd"]**4
    P_in_cd = (B1 * B2_cd * (mean_cd_water_flow / 1e3) * p["L_cd"])**(1.0/(4.0*tc + 1.0))
    P_in_cd_mmHg = (P_in_cd + P_interstitial) / mmHg_conv

    B2_dt = Pc_dt**(4.0*tc) / p["Dc_dt"]**4
    P_in_dt = (P_in_cd**(4.0*tc + 1.0) + B1 * B2_dt * (water_in_DCT / 1e3) * p["L_dct"])**(1.0/(4.0*tc + 1.0))

    B2_lh_asc = Pc_lh_asc**(4.0*tc) / p["Dc_lh"]**4
    P_in_lh_asc = (P_in_dt**(4.0*tc + 1.0) + B1 * B2_lh_asc * (water_in_AscLoH / 1e3) * p["L_lh_asc"])**(1.0/(4.0*tc + 1.0))

    A_lh_des = effective_AscLoH_Reab_Rate / (water_in_DescLoH * Na_concentration_in_DescLoH)
    B2_lh_des = (Pc_lh_des**(4.0*tc)) * (water_in_DescLoH / 1e3) / (p["Dc_lh"]**4 * A_lh_des)
    P_in_lh_des = (P_in_lh_asc**(4.0*tc + 1.0) + B1 * B2_lh_des * (1.0 - math.exp(-A_lh_des * p["L_lh_des"])))**(1.0/(4.0*tc + 1.0))

    A_na = Na_reabs_per_unit_length
    flow_integral_pt = (SN_filtered_Na_load / A_na) * (1.0 - math.exp(-A_na * L_pt))
    B2_pt = Pc_pt_Pa**(4.0*tc) / p["Dc_pt_nom"]**4
    B3_pt = (SNGFR_nL_min / 1e12) / SN_filtered_Na_load
    P_in_pt = (P_in_lh_des**(4.0*tc + 1.0) + B1 * B2_pt * B3_pt * flow_integral_pt)**(1.0/(4.0*tc + 1.0))
    P_in_pt_mmHg = (P_in_pt + P_interstitial) / mmHg_conv

    # =========================================================
    # RAAS
    # =========================================================

    # Aldosterone
    AT1_aldo_int = 1.0 - p["AT1_aldo_slope"] * p["nominal_equilibrium_AT1_bound_AngII"]
    AngII_effect_on_aldo = AT1_aldo_int + p["AT1_aldo_slope"] * AT1_bound_AngII
    normalized_aldosterone_level = p["K_Na_ratio_effect_on_aldo"] * AngII_effect_on_aldo

    # Renin secretion
    # md_effect uses delayed version (SN_macula_densa_Na_flow_delayed)
    md_effect_on_renin_secretion = p["md_renin_A"] * math.exp(
        -p["md_renin_tau"]
        * (SN_macula_densa_Na_flow_delayed * neph - p["nom_LoH_Na_outflow"])
    )

    _at1_ratio = max(AT1_bound_AngII / p["nominal_equilibrium_AT1_bound_AngII"], 1e-15)
    AT1_bound_AngII_effect_on_PRA = 10.0 ** (
        p["AT1_PRC_slope"] * math.log10(_at1_ratio)
        + p["AT1_PRC_yint"]
    )

    aldo_renin_intercept = 1.0 - p["aldo_renin_slope"]
    aldo_effect_on_renin_secretion = (
        p["aldo_renin_slope"] * normalized_aldosterone_level_delayed + aldo_renin_intercept
    )

    plasma_renin_activity = (
        p["concentration_to_renin_activity_conversion_plasma"]
        * plasma_renin_concentration
        * p["DRI_effect_on_PRA"]
    )

    renin_secretion_rate = (
        (math.log(2.0) / p["renin_half_life"])
        * p["nominal_equilibrium_PRC"]
        * AT1_bound_AngII_effect_on_PRA
        * md_effect_on_renin_secretion
        * p["HCTZ_effect_on_renin_secretion"]
        * aldo_effect_on_renin_secretion
    )

    renin_degradation_rate        = math.log(2.0) / p["renin_half_life"]
    AngI_degradation_rate         = math.log(2.0) / p["AngI_half_life"]
    AngII_degradation_rate        = math.log(2.0) / p["AngII_half_life"]
    AT1_bound_AngII_degradation_rate = math.log(2.0) / p["AT1_bound_AngII_half_life"]
    AT2_bound_AngII_degradation_rate = math.log(2.0) / p["AT2_bound_AngII_half_life"]

    ACE_activity  = p["nominal_ACE_activity"] * (1.0 - p["pct_target_inhibition_ACEi"])
    chymase_activity = p["nominal_chymase_activity"]
    AT1_receptor_binding_rate = p["nominal_AT1_receptor_binding_rate"] * (1.0 - p["pct_target_inhibition_ARB"])
    AT2_receptor_binding_rate = p["nominal_AT2_receptor_binding_rate"]

    # =========================================================
    # ODEs
    # =========================================================

    # RAAS
    dAngI = (
        plasma_renin_activity
        - AngI * (chymase_activity + ACE_activity)
        - AngI * AngI_degradation_rate
    )
    dAngII = (
        AngI * (chymase_activity + ACE_activity)
        - AngII * AngII_degradation_rate
        - AngII * AT1_receptor_binding_rate
        - AngII * AT2_receptor_binding_rate
    )
    dAT1_bound_AngII = (
        AngII * AT1_receptor_binding_rate
        - AT1_bound_AngII_degradation_rate * AT1_bound_AngII
    )
    dAT2_bound_AngII = (
        AngII * AT2_receptor_binding_rate
        - AT2_bound_AngII_degradation_rate * AT2_bound_AngII
    )
    dplasma_renin_concentration = (
        renin_secretion_rate - plasma_renin_concentration * renin_degradation_rate
    )

    # Blood volume / ECF
    dblood_volume_L = (
        p["C_water_intake_ecf_volume"] * water_intake
        + p["C_urine_flow_ecf_volume"] * urine_flow_rate
        + p["Q_water"] * (Na_concentration - ECF_Na_concentration)
    )
    dextracellular_fluid_volume = p["Q_water"] * (ECF_Na_concentration - Na_concentration)

    # Sodium
    dsodium_amount = (
        p["C_na_excretion_na_amount"] * Na_excretion_via_urine
        + p["C_na_intake_na_amount"] * p["Na_intake_rate"]
        + p["Q_Na"] * (ECF_Na_concentration - Na_concentration)
    )
    dECF_sodium_amount = p["Q_Na"] * (Na_concentration - ECF_Na_concentration)

    # Fast ODE delay states
    dtubulo_glomerular_feedback_effect = (
        p["C_tgf"] * (tubulo_glomerular_feedback_signal - tubulo_glomerular_feedback_effect)
    )
    dnormalized_aldosterone_level_delayed = (
        p["C_aldo_secretion"] * (normalized_aldosterone_level - normalized_aldosterone_level_delayed)
    )
    dpreafferent_pressure_autoreg_signal = (
        500.0 * (preafferent_pressure_autoreg_function - preafferent_pressure_autoreg_signal)
    )
    dglomerular_pressure_autoreg_signal = (
        500.0 * (glomerular_pressure_autoreg_function - glomerular_pressure_autoreg_signal)
    )
    dcardiac_output_delayed = p["C_cardiac_output_delayed"] * (cardiac_output - cardiac_output_delayed)

    # PI controller error signals
    dCO_error           = p["C_co_error"] * (cardiac_output - p["CO_nom"])
    dNa_concentration_error = p["C_Na_error"] * (Na_concentration - p["ref_Na_concentration"])

    # Vasopressin delay
    dnormalized_vasopressin_concentration_delayed = (
        p["C_vasopressin_delay"]
        * (normalized_vasopressin_concentration - normalized_vasopressin_concentration_delayed)
    )

    # TGF resetting (C_tgf_reset=0 → effectively frozen)
    dF0_TGF = p["C_tgf_reset"] * (SN_macula_densa_Na_flow * neph - F0_TGF)

    # P_bowmans (fast relaxation to P_in_pt_mmHg)
    dP_bowmans = p["C_P_bowmans"] * (P_in_pt_mmHg - P_bowmans)

    # Oncotic pressure difference (fast relaxation)
    doncotic_pressure_difference = p["C_P_oncotic"] * (oncotic_pressure_avg - oncotic_pressure_difference)

    # MD Na flow delay
    dSN_macula_densa_Na_flow_delayed = (
        p["C_md_flow"] * (SN_macula_densa_Na_flow - SN_macula_densa_Na_flow_delayed)
    )

    # Serum creatinine
    dserum_creatinine = p["creatinine_synthesis_rate"] - creatinine_clearance_rate

    return [
        dAngI,
        dAngII,
        dAT1_bound_AngII,
        dAT2_bound_AngII,
        dplasma_renin_concentration,
        dblood_volume_L,
        dextracellular_fluid_volume,
        dsodium_amount,
        dECF_sodium_amount,
        dtubulo_glomerular_feedback_effect,
        dnormalized_aldosterone_level_delayed,
        dpreafferent_pressure_autoreg_signal,
        dglomerular_pressure_autoreg_signal,
        dcardiac_output_delayed,
        dCO_error,
        dNa_concentration_error,
        dnormalized_vasopressin_concentration_delayed,
        dF0_TGF,
        dP_bowmans,
        doncotic_pressure_difference,
        dSN_macula_densa_Na_flow_delayed,
        dserum_creatinine,
    ]


# ---------------------------------------------------------------------------
# Derived outputs (same algebra as model, given a state vector + params)
# ---------------------------------------------------------------------------

def compute_outputs(y, p):
    """Compute key derived quantities from a state vector."""
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
     serum_creatinine) = y

    neph = p["baseline_nephrons"]
    L_m3 = p["L_m3"]
    visc  = p["viscosity_length_constant"]

    nom_aff_d = p["nom_afferent_diameter"]
    nom_eff_d = p["nom_efferent_diameter"]
    pi    = p["pi"]
    gamma = p["gamma"]
    mmHg_conv = p["mmHg_Nperm2_conv"]

    # Systemic
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
    MAP = cardiac_output * total_peripheral_resistance

    # Renal
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

    preaff_arteriole_signal_multiplier = (
        AT1_effect_on_preaff * preafferent_pressure_autoreg_signal
        * p["CCB_effect_on_preafferent_resistance"]
    )
    preaff_arteriole_adjusted_signal_multiplier = (
        _sigmoid(p["preaff_signal_nonlin_scale"] * (1.0 - preaff_arteriole_signal_multiplier)) + 0.5
    )
    preafferent_arteriole_resistance = (
        p["nom_preafferent_arteriole_resistance"] * preaff_arteriole_adjusted_signal_multiplier
    )

    nom_afferent_arteriole_resistance = L_m3 * visc / nom_aff_d**4
    nom_efferent_arteriole_resistance = L_m3 * visc / nom_eff_d**4

    afferent_arteriole_signal_multiplier = (
        tubulo_glomerular_feedback_effect * AT1_effect_on_aff
        * glomerular_pressure_autoreg_signal * p["CCB_effect_on_afferent_resistance"]
    )
    afferent_arteriole_adjusted_signal_multiplier = (
        _sigmoid(p["afferent_signal_nonlin_scale"] * (1.0 - afferent_arteriole_signal_multiplier)) + 0.5
    )
    afferent_arteriole_resistance = (
        nom_afferent_arteriole_resistance * afferent_arteriole_adjusted_signal_multiplier
    )

    efferent_arteriole_signal_multiplier = AT1_effect_on_eff * p["CCB_effect_on_efferent_resistance"]
    efferent_arteriole_adjusted_signal_multiplier = (
        _sigmoid(p["efferent_signal_nonlin_scale"] * (1.0 - efferent_arteriole_signal_multiplier)) + 0.5
    )
    efferent_arteriole_resistance = (
        nom_efferent_arteriole_resistance * efferent_arteriole_adjusted_signal_multiplier
    )

    peritubular_resistance = p["nom_peritubular_resistance"] * neph
    renal_vascular_resistance = (
        preafferent_arteriole_resistance
        + (afferent_arteriole_resistance + efferent_arteriole_resistance + peritubular_resistance) / neph
    )
    renal_blood_flow_L_min = (MAP - p["P_venous"]) / renal_vascular_resistance

    glomerular_pressure_val = (
        MAP
        - renal_blood_flow_L_min
        * (preafferent_arteriole_resistance + afferent_arteriole_resistance / neph)
    )

    SNGFR_nL_min = p["nom_Kf"] * (glomerular_pressure_val - oncotic_pressure_difference - P_bowmans)
    GFR_L   = SNGFR_nL_min / 1e6 / 1000.0 * neph
    GFR_mL  = GFR_L * 1000.0

    Na_concentration = sodium_amount / blood_volume_L
    serum_creatinine_conc = serum_creatinine / blood_volume_L  # mg/dL (state stored as mg/dL × L)
    creatinine_clearance  = GFR_mL * p["dl_ml"] * serum_creatinine_conc  # mg/min

    return {
        "MAP":                    MAP,
        "GFR_ml_min":             GFR_mL,
        "renal_blood_flow_L_min": renal_blood_flow_L_min,
        "glomerular_pressure":    glomerular_pressure_val,
        "Na_concentration":       Na_concentration,
        "blood_volume_L":         blood_volume_L,
        "AngI":                   AngI,
        "AngII":                  AngII,
        "AT1_bound_AngII":        AT1_bound_AngII,
        "plasma_renin_concentration": plasma_renin_concentration,
        # In R, serum_creatinine state = eq_cr[mg/dL] * BV[L], so state/BV = mg/dL directly
        "serum_creatinine_mg_dL": serum_creatinine / blood_volume_L,  # mg/dL
        "cardiac_output":         cardiac_output,
    }


# ---------------------------------------------------------------------------
# Run baseline steady-state simulation
# ---------------------------------------------------------------------------

def run_baseline(p=None):
    """
    Solve to steady state by integrating for 100 000 time units (matching R runModel.R).
    Returns a dict of key outputs.
    """
    if p is None:
        p = make_params()

    y0 = initial_conditions(p)

    # First long integration (R does seq(0,100000,by=10))
    sol = solve_ivp(
        fun=lambda t, y: odes(t, y, p),
        t_span=(0.0, 100_000.0),
        y0=y0,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
        max_step=100.0,
    )

    if not sol.success:
        raise RuntimeError(f"Solver failed in first pass: {sol.message}")

    # Use final state as new ICs and run again (as in R)
    y1 = list(sol.y[:, -1])
    sol2 = solve_ivp(
        fun=lambda t, y: odes(t, y, p),
        t_span=(0.0, 100_000.0),
        y0=y1,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
        max_step=100.0,
    )

    if not sol2.success:
        raise RuntimeError(f"Solver failed in second pass: {sol2.message}")

    y_ss = list(sol2.y[:, -1])
    return compute_outputs(y_ss, p)


def run_arb(dose, p=None):
    """
    Run with ARB (AT1 receptor blockade).
    dose: fractional inhibition of AT1 receptor binding (0–1).
    """
    if p is None:
        p = make_params()
    p_drug = dict(p)
    p_drug["pct_target_inhibition_ARB"] = float(dose)
    return run_baseline(p=p_drug)


# ---------------------------------------------------------------------------
# Quick CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running baseline simulation …")
    p = make_params()
    out = run_baseline(p)
    print("\nBaseline outputs:")
    for k, v in out.items():
        print(f"  {k:35s} = {v:.4f}")
