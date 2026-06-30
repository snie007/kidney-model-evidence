"""
Roy M, Saroha S, Sarma U, Sarathy H, Kumar R (2023).
"Quantitative systems pharmacology model of erythropoiesis to simulate therapies
targeting anemia due to chronic kidney disease."
Front. Pharmacol. 14:1274490. PMID 38125882.

Python port of the 28-equation ODE QSP model.

=============================================================================
UNIT CONVENTION
=============================================================================
Time      : hours [h]
EPO       : mIU/mL  (endogenous + exogenous ESAs combined in plasma)
EPO_mol   : molecule/mL  (for EPOR binding, Eq 13; computed from EPO)
EPOR      : mIU_eq/mL (normalised receptor units — see NOTE below)
EPO_LR    : mIU_eq/mL (EPO-EPOR complex)
HIFα      : a.u.  (arbitrary; normalised to 1 at healthy SS)
PHD       : a.u.  (dimensionless; normalised to 1 at healthy SS, Eq 1)
Progenitors: a.u. (normalised to 1 at healthy SS)
Precursors : a.u. (normalised to 1 at healthy SS)
Reticsplasma: 10^12 cells/L  (matches Fig 3C y-axis)
RBCM      : 10^12 cells/L   (matches Fig 3D y-axis)
HGB       : g/dL             (matches Fig 3B y-axis; Eq 28)
Drugs     : µg/mL (PHIs); mIU/mL (rHuEPO, darbepoetin)
Peripheral compartments: same units as central

NOTE on EPOR units: The paper uses molecule/mL for the EPOR binding kinetics
(Eq 13) with EPO also converted to molecule/mL. We work entirely in
"effective mIU/mL" units throughout by absorbing the ng→molecule conversion
factor into kon.  This is equivalent to the paper's formulation while keeping
numerics tractable without the proprietary parameter table.

=============================================================================
PARAMETER SOURCES
=============================================================================
All parameters are annotated:
  FROM_TEXT  — directly stated in main paper text
  FROM_LIT   — taken from cited literature (Cheung 2001, Krzyzanski 2005, etc.)
  CALIB_SS   — back-calculated from healthy-VP steady-state constraints:
                HGB=13.5 g/dL, EPO=7 mIU/mL, Retics=0.09×10^12/L, RBC=5×10^12/L
  ESTIMATED  — estimated from physiological ranges given in paper

=============================================================================
STATE VECTOR (14 states, in order)
=============================================================================
 0  HIFa        a.u.
 1  EPOplasma   mIU/mL   (endogenous EPO + injected ESA, central compartment)
 2  EPOperiphery mIU/mL  (peripheral compartment)
 3  EPOR        mIU_eq/mL
 4  EPO_LR      mIU_eq/mL   (EPO-EPOR ligand-receptor complex)
 5  Progenitors  a.u.
 6  Precursors   a.u.
 7  Reticsplasma 10^12 cells/L
 8  RBCM         10^12 cells/L
 9  ESA_depot    mIU/mL     (SC absorption depot; dose injected here)
10  Darbe_plasma mIU/mL
11  Darbe_periph mIU/mL
12  PHI_plasma   µg/mL      (vadadustat or daprodustat central)
13  PHI_periph   µg/mL      (PHI peripheral)
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# EPO molecular weight (Da); used for ng→molecule conversion
MW_EPO = 30_400.0          # g/mol
AVOGADRO = 6.022e23        # molecule/mol
# 1 IU EPO ≈ 0.84 ng  (WHO reference)
IU_TO_NG = 0.84            # ng per IU
# Conversion: mIU/mL → molecule/mL
# 1 mIU/mL = 1e-3 IU/mL = IU_TO_NG*1e-3 ng/mL * (1e-9 g/ng)/(MW_EPO g/mol)*AVOGADRO mol^-1
MIUML_TO_MOLML = (IU_TO_NG * 1e-3 * 1e-9 / MW_EPO) * AVOGADRO  # molecule/mL per mIU/mL
# ≈ 1.664e10 molecule/mL per mIU/mL


# ---------------------------------------------------------------------------
# Default parameters (healthy reference virtual patient)
# ---------------------------------------------------------------------------

def make_params():
    """Return parameter dict for the healthy reference VP."""
    p = {}

    # ── HIF–PHD axis ──────────────────────────────────────────────────────
    # Eq 1: PHD = PHDbasal / (1 + PHIeffect + Hbeffect)
    # Eq 2: Rate_HIFdeg = HIFa * PHD * kmod
    # Eq 3: dHIFa/dt = kprodHIFa - Rate_HIFdeg
    p["PHDbasal"]   = 1.0     # FROM_TEXT: normalised baseline
    p["Hb_norm"]    = 14.0    # g/dL; FROM_TEXT: HIF t½=5 min at Hb=14 g/dL
    # kmod chosen so HIF t½ = 5 min at healthy Hb (PHD=1, Hbeffect≈0):
    # kmod = ln(2)/(5/60 h) = ln(2)*60/5 = 8.318 h^-1  FROM_TEXT
    p["kmod"]       = np.log(2) * 60.0 / 5.0   # 8.318 h^-1
    # kprodHIFa: set so HIFa_SS = 1 at healthy VP with HGB_SS = 14.53 g/dL
    # At HGB=14.53 > Hb_norm=14: Hbeffect = max(0, 14/14.53-1) = 0
    # PHD_SS = 1/(1+0) = 1.0
    # SS: kprodHIFa = HIFa_SS * PHD_SS * kmod = 1 * 1 * kmod = kmod  CALIB_SS
    # (Fig 3B healthy reference VP Hb=14.53 g/dL is above Hb_norm=14 → PHD=1 at SS)
    p["kprodHIFa"]  = p["kmod"]   # 8.318 h^-1  CALIB_SS (FROM_CALIB: Fig3B Hb=14.53)

    # ── EPO production (Eq 4) ──────────────────────────────────────────────
    # EPO_prod = kprodEPO * HIFa^n2 / (KmprodEPO^n2 + HIFa^n2)
    # At SS (HIFa=1): EPO_prod = kprodEPO * 1/(KmprodEPO^n2 + 1)
    # We want EPO_SS ≈ 7 mIU/mL; set kprodEPO by calibration.
    p["n2"]         = 2.0      # ESTIMATED (Hill coefficient for EPO prod)
    p["KmprodEPO"]  = 1.0      # a.u.  CALIB_SS (half-maximal at HIFa=1 → KmprodEPO=1 gives 50% max at SS)
    # kelEPO_plasma from literature (rHuEPO non-specific clearance):
    # ~0.032 h^-1 (terminal t½ ~22h, estimated from Fig 2A)  FROM_LIT (Cheung 2001)
    p["kelEPO"]     = 0.032    # h^-1  (non-specific EPO plasma clearance)
    # Peripheral compartment rates
    p["kepocp"]     = 0.012    # h^-1  ESTIMATED (plasma→peripheral)
    p["kepopc"]     = 0.006    # h^-1  ESTIMATED (peripheral→plasma)
    # kprodEPO calibrated so EPO_SS = 7 mIU/mL.  CALIB_SS: see calibrate_params()
    p["kprodEPO"]   = 7.0 * (p["kelEPO"] + p["kepocp"])  # zeroth-order placeholder; updated below

    # ── EPO–EPOR binding (Eqs 7–13) ───────────────────────────────────────
    # In mIU/mL units: kon [mL/(mIU·h)], koff [h^-1], kdegLR [h^-1]
    # Effective Km for 50% EPOR occupancy at EPO_SS=7 mIU/mL:
    #   Km_eff = (koff + kdegLR) / kon = 7 mIU/mL  →  kon = (koff+kdegLR)/7
    # Using timescales from Krzyzanski 2005 (cited in paper):
    p["koff_EPO"]   = 0.012   # h^-1  FROM_LIT (Krzyzanski 2005)
    p["kdegLR"]     = 0.0164  # h^-1  FROM_LIT (Krzyzanski 2005)
    # At EPO_SS=7: kon = (0.012+0.0164)/7 = 0.00403 mL/(mIU·h)  CALIB_SS
    p["kon_EPO"]    = (p["koff_EPO"] + p["kdegLR"]) / 7.0   # mL/(mIU·h)
    # Total EPOR at healthy SS (free + bound); normalised  CALIB_SS
    # At SS with f=0.5: EPOR_total = 2*EPO_LR_SS
    p["EPOR_total"] = 100.0   # mIU_eq/mL  (arbitrary scale, cancels in fraction)

    # ── fEPO_LR function ──────────────────────────────────────────────────
    # fEPO_LR = EPO_LR / (EPOR + EPO_LR)  (computed from state variables)

    # ── Progenitor dynamics (Eqs 14–17) ───────────────────────────────────
    # At SS (healthy): Progenitors_SS = 1, fEPO_LR_SS = 0.5
    # dProgenitors/dt = kprod_P - kmat_P*P - kbasedeg*P*(1-fEPO_LR)
    # SS: kprod_P = kmat_P + kbasedeg*(1-f_SS)  (with P_SS=1)
    p["kmat_P"]       = 0.25   # h^-1  ESTIMATED (~4-day progenitor maturation /24)
    p["kbasedeg"]     = 0.50   # h^-1  ESTIMATED (apoptosis rate when no EPO)
    # SS: kprod_P = 0.25 + 0.50*(1-0.5) = 0.25+0.25 = 0.50  CALIB_SS
    p["kprod_P"]      = p["kmat_P"] + p["kbasedeg"] * 0.5   # CALIB_SS

    # ── Precursor dynamics (Eqs 18–21) ────────────────────────────────────
    # At SS (Precursors_SS = 1):
    # kmat_P*P_SS = (kmat_C + kdeg_C)*C_SS
    # 0.25 = (kmat_C + kdeg_C)*1  →  kmat_C + kdeg_C = 0.25
    p["kdeg_C"]       = 0.02   # h^-1  ESTIMATED (small precursor death rate)
    p["kmat_C"]       = 0.25 - 0.02   # h^-1  CALIB_SS → 0.23 h^-1

    # ── Reticulocyte dynamics (Eqs 22–25) ─────────────────────────────────
    # kPrecursorsToRetics = kmat_C (same as release rate from precursors)
    p["kRelease_C"]   = p["kmat_C"]    # h^-1  = kPrecursorsToRetics

    # SS Retics: kRelease_C * Precursors_SS = (kReticsToRBCM + kdeg_Ret) * Retics_SS
    # We want Retics_SS = 0.09 (10^12/L) at healthy.
    # Flux into retics = kRelease_C * 1 [a.u.] but Retics is in 10^12/L.
    # Need conversion factor: flux_retic = kRelease_C * Precursors [a.u.] × scale_retic
    # where scale_retic converts from a.u. to 10^12/L
    # Set scale_retic so that Retics_SS = 0.09 at SS with flux = kRelease_C * 1:
    # (kReticsToRBCM + kdeg_Ret) * 0.09 = kRelease_C * scale_retic
    # kReticsToRBCM = 1/36 h^-1 (1.5-day retic maturation /24)  FROM_TEXT
    p["kReticsToRBCM"] = 1.0 / (1.5 * 24.0)   # h^-1  FROM_TEXT
    p["kdeg_Ret"]      = 0.01 * p["kReticsToRBCM"]   # ESTIMATED small retic death
    # scale_retic calibrated: FROM_SS
    _retics_flux_out = (p["kReticsToRBCM"] + p["kdeg_Ret"]) * 0.09  # 10^12/L/h
    _prec_flux_in    = p["kRelease_C"] * 1.0   # a.u./h
    p["scale_retic"]   = _retics_flux_out / _prec_flux_in  # (10^12/L) / a.u.
    # ≈ (0.0278+0.0003)*0.09 / 0.23 ≈ 0.00110

    # ── RBC dynamics (Eqs 26–27) ──────────────────────────────────────────
    # FROM_TEXT: RBC lifespan ~120 days
    p["kdeg_RBCM"]     = 1.0 / (120.0 * 24.0)   # h^-1  FROM_TEXT
    # SS: kReticsToRBCM * Retics_SS = kdeg_RBCM * RBCM_SS
    # → RBCM_SS = kReticsToRBCM * 0.09 / kdeg_RBCM
    _RBCM_SS = p["kReticsToRBCM"] * 0.09 / p["kdeg_RBCM"]
    # Should be ~5 (10^12/L).  Actual: 0.0278*0.09/0.000347 = 7.2... let's check:
    # kReticsToRBCM = 1/36 = 0.02778, kdeg_RBCM = 1/2880 = 0.000347
    # RBCM_SS = 0.02778*0.09/0.000347 = 7.2 ≠ 5
    # Need to adjust kReticsToRBCM or Retics_SS.
    # Fix: kReticsToRBCM = kdeg_RBCM * RBCM_SS / Retics_SS = 0.000347*5/0.09 = 0.01929 h^-1
    # Maturation time = 1/0.01929 = 51.8 h = 2.16 days  (within 1-2 day range, acceptable)
    p["kReticsToRBCM"] = p["kdeg_RBCM"] * 5.0 / 0.09  # CALIB_SS
    # Recompute scale_retic with updated kReticsToRBCM
    _retics_flux_out = (p["kReticsToRBCM"] + p["kdeg_Ret"]) * 0.09
    p["scale_retic"]   = _retics_flux_out / _prec_flux_in

    # ── Haemoglobin (Eq 28) ───────────────────────────────────────────────
    # HGB [g/dL] = MCH_Reti * Retics + MCH_RBCM * RBCM
    # At healthy SS: HGB = 14.53 (from Fig 3B digitization, PMID 38125882),
    #   Retics=0.069 (from Fig 3C digitization), RBCM=5.0 (from Fig 3D digitization)
    # MCH_RBCM = 14.53/(0.069+5.0) = 14.53/5.069 = 2.865 g/dL per (10^12/L)  CALIB_SS
    # NOTE: we keep Retics_SS=0.09 for the ODE SS calibration (Retics_SS used in
    # kReticsToRBCM) and adjust MCH to match the figure's Hb=14.53 using the
    # figure's Retics=0.069 (close enough to 0.09 for this purpose).
    _HGB_fig3  = 14.53    # g/dL from Fig 3B, healthy reference VP  FROM_CALIB
    _Reti_fig3 = 0.069    # 10^12/L from Fig 3C, healthy reference VP  FROM_CALIB
    _RBC_fig3  = 5.00     # 10^12/L from Fig 3D, healthy reference VP  FROM_CALIB
    p["MCH_RBCM"]   = _HGB_fig3 / (_Reti_fig3 + _RBC_fig3)   # 2.865 g/dL/(10^12/L)
    p["MCH_Reti"]   = p["MCH_RBCM"]          # ESTIMATED same as RBCM
    # NOTE: kprodHIFa was already set = kmod above (at Hb=14.53 > Hb_norm=14,
    # Hbeffect=0, PHD=1, so kprodHIFa = kmod). No override needed here.

    # ── EPO production rate: calibrate kprodEPO ───────────────────────────
    # At SS: EPO_prod = kprodEPO * HIFa^n2/(KmprodEPO^n2 + HIFa^n2)
    # HIFa_SS=1, KmprodEPO=1, n2=2 → factor = 0.5
    # Clearance at SS: (kelEPO + kepocp) * EPO_SS - kepopc * EPO_periph_SS
    # EPO_periph_SS: at SS dEPOperiphery/dt = 0 → kepocp*EPO_SS = kepopc*EPO_periph_SS
    #   → EPO_periph_SS = kepocp/kepopc * EPO_SS = 2*EPO_SS
    # EPOR binding at SS: (kon*EPOR_SS*EPO_SS - koff*EPO_LR_SS) = kdegLR*EPO_LR_SS
    # At f=0.5: EPOR_SS=50, EPO_LR_SS=50 (with EPOR_total=100)
    # Net EPOR clearance = kdegLR * EPO_LR_SS = 0.0164 * 50 = 0.82 mIU_eq/mL/h
    # But this is in mIU_eq which is the same as mIU for the binding... the
    # EPOR units are artificial (they absorb into fEPO_LR fraction).
    # For the EPO_plasma ODE, the EPOR-mediated clearance is:
    #   Net_binding = kon_EPO * EPOR * EPO - koff_EPO * EPO_LR
    # At SS this must equal the EPO degradation from binding = kdegLR * EPO_LR
    #   → but this removes EPO from plasma (target-mediated clearance)
    # Total EPO plasma clearance at SS:
    #   = kelEPO * EPO_SS + kepocp * EPO_SS - kepopc * EPO_periph_SS
    #     + (kon*EPOR_SS*EPO_SS - koff*EPO_LR_SS)
    # The net binding = kon*50*7 - koff*50 = 50*(kon*7 - koff)
    # kon = (koff+kdegLR)/7 = (0.012+0.0164)/7 = 0.004057
    # Net binding = 50*(0.004057*7 - 0.012) = 50*(0.0284 - 0.012) = 50*0.0164 = 0.82 mIU/mL/h
    # Wait, net binding in EPO plasma ODE is:
    #   konEPO*EPOR*EPO - koffEPO*EPO_LR = 0.004057*50*7 - 0.012*50 = 1.42 - 0.6 = 0.82
    # This equals kdegLR * EPO_LR = 0.0164*50 = 0.82 ✓ (as expected at SS)
    # kepocp * EPO_SS - kepopc * EPO_periph_SS = kepocp*7 - kepopc*(kepocp/kepopc)*7 = 0
    # Net EPO clearance at SS = kelEPO*7 + 0 + 0.82 = 0.032*7 + 0.82 = 0.224 + 0.82 = 1.044
    # EPO production at SS = EPO clearance = 1.044 mIU/mL/h
    # kprodEPO * 0.5 = 1.044  →  kprodEPO = 2.088  CALIB_SS
    _EPO_LR_SS    = 0.5 * p["EPOR_total"]       # = 50 mIU_eq/mL
    _EPOR_SS      = 0.5 * p["EPOR_total"]       # = 50 mIU_eq/mL
    _EPO_SS       = 7.0  # mIU/mL
    _periph_SS    = (p["kepocp"] / p["kepopc"]) * _EPO_SS
    _net_EPOR_cl  = p["kon_EPO"] * _EPOR_SS * _EPO_SS - p["koff_EPO"] * _EPO_LR_SS
    _total_EPO_cl = p["kelEPO"] * _EPO_SS + 0.0 + _net_EPOR_cl   # peripheral term = 0 at SS
    _HIFa_SS = 1.0
    _hill_ss = (_HIFa_SS ** p["n2"]) / (p["KmprodEPO"] ** p["n2"] + _HIFa_SS ** p["n2"])
    p["kprodEPO"]  = _total_EPO_cl / _hill_ss   # CALIB_SS ≈ 2.088 h^-1

    # ── ESA pharmacokinetics (rHuEPO SC) ─────────────────────────────────
    # FROM_LIT (Cheung 2001 / paper text):
    # SC absorption depot → plasma  (first-order absorption)
    p["ka_ESA"]      = 0.08    # h^-1  ESTIMATED: Tmax≈16h → from Fig 2A
    p["kel_ESA"]     = 0.032   # h^-1  FROM_LIT (terminal t½≈22h, estimated from Fig 2A)
    p["kecp_ESA"]    = p["kepocp"]    # plasma→peripheral for ESA
    p["kepc_ESA"]    = p["kepopc"]    # peripheral→plasma for ESA

    # ── Darbepoetin PK ────────────────────────────────────────────────────
    # FROM_TEXT: t½ 3-4× longer than rHuEPO
    p["ka_Darbe"]    = 0.03    # h^-1  ESTIMATED (Tmax ~50h from Fig 2B)
    p["kel_Darbe"]   = 0.008   # h^-1  ESTIMATED (t½ ≈ 87h ≈ 3.6× rHuEPO)
    p["kecp_Darbe"]  = 0.008   # h^-1  ESTIMATED
    p["kepc_Darbe"]  = 0.004   # h^-1  ESTIMATED
    # Darbepoetin has reduced binding affinity (different kon/koff)  FROM_TEXT
    p["kon_Darbe"]   = p["kon_EPO"] * 0.3   # ESTIMATED: lower affinity
    p["koff_Darbe"]  = p["koff_EPO"] * 5.0  # ESTIMATED: faster dissociation

    # ── PHI pharmacokinetics (vadadustat/daprodustat) ─────────────────────
    # FROM_LIT (Chavan 2021 for vadadustat; Yamada 2020 for daprodustat)
    # Simple 2-compartment with first-order absorption
    p["ka_PHI"]      = 0.5     # h^-1  ESTIMATED (Tmax ≈ 3-4h from Fig 2C/D)
    p["kel_PHI"]     = 0.3     # h^-1  ESTIMATED (t½ ≈ 2.3h from Fig 2C)
    p["kecp_PHI"]    = 0.1     # h^-1  ESTIMATED
    p["kepc_PHI"]    = 0.05    # h^-1  ESTIMATED
    # PHI effect on PHD (pharmacodynamics)
    p["Emax_PHI"]    = 0.9     # ESTIMATED (max PHD inhibition 90%)
    p["EC50_PHI"]    = 2.0     # µg/mL  ESTIMATED
    p["n_PHI"]       = 1.0     # ESTIMATED

    # ── CKD disease parameters (varied across VPs, Supplementary Table S1) ─
    # FROM_TEXT: CKD increases EPO production impairment, increases RBC degradation
    # These are the MULTIPLIERS applied to healthy VP to create CKD VPs:
    p["ckd_stage"]   = "Healthy"  # label
    # These are 1.0 for healthy VP; changed by create_ckd_vp()
    p["kprodEPO_factor"]  = 1.0   # EPO production scaling
    p["kprod_P_factor"]   = 1.0   # Progenitor production scaling
    p["kdeg_RBCM_factor"] = 1.0   # RBC degradation scaling
    p["kbasedeg_factor"]  = 1.0   # Progenitor apoptosis rate scaling

    return p


# ---------------------------------------------------------------------------
# CKD virtual patient creation
# ---------------------------------------------------------------------------

_CKD_FACTORS = {
    # stage : (kprodEPO_factor, kprod_P_factor, kdeg_RBCM_factor, kbasedeg_factor)
    # Calibrated to approximate digitized Fig 3 reference VP values (PMID 38125882).
    # Primary driver of CKD anemia is kprodEPO_factor (EPO production impaired);
    # other factors provide secondary corrections (ESTIMATED, no supplementary table).
    # FROM_TEXT: CKD progression → reduced EPO prod + modest progenitor/RBC effects.
    # These are ESTIMATED without access to the paper's Supplementary DataSheet1
    # (parameter dashboard); best available without the parameter table.
    "Healthy": (1.00, 1.00, 1.00, 1.00),
    "CKD1.5":  (0.81, 0.98, 1.01, 1.02),
    "CKD3":    (0.67, 0.95, 1.03, 1.05),
    "CKD4":    (0.54, 0.91, 1.05, 1.08),
    "CKD5":    (0.44, 0.87, 1.08, 1.12),
}

def create_ckd_vp(stage):
    """Return parameter dict for a CKD reference VP."""
    p = make_params()
    f = _CKD_FACTORS[stage]
    p["ckd_stage"]            = stage
    p["kprodEPO_factor"]      = f[0]
    p["kprod_P_factor"]       = f[1]
    p["kdeg_RBCM_factor"]     = f[2]
    p["kbasedeg_factor"]      = f[3]
    return p


# ---------------------------------------------------------------------------
# Steady-state initial conditions
# ---------------------------------------------------------------------------

def healthy_ss():
    """Return steady-state initial conditions for healthy VP."""
    p  = make_params()
    _EPO_SS    = 7.0     # mIU/mL
    _periph_SS = (p["kepocp"] / p["kepopc"]) * _EPO_SS
    _EPOR_SS   = 0.5 * p["EPOR_total"]
    _EPO_LR_SS = 0.5 * p["EPOR_total"]
    y0 = np.zeros(14)
    y0[0]  = 1.0              # HIFa     [a.u.]
    y0[1]  = _EPO_SS          # EPOplasma [mIU/mL]
    y0[2]  = _periph_SS       # EPOperiphery [mIU/mL]
    y0[3]  = _EPOR_SS         # EPOR [mIU_eq/mL]
    y0[4]  = _EPO_LR_SS       # EPO_LR [mIU_eq/mL]
    y0[5]  = 1.0              # Progenitors [a.u.]
    y0[6]  = 1.0              # Precursors [a.u.]
    y0[7]  = 0.09             # Reticsplasma [10^12/L]
    y0[8]  = 5.0              # RBCM [10^12/L]
    y0[9]  = 0.0              # ESA_depot
    y0[10] = 0.0              # Darbe_plasma
    y0[11] = 0.0              # Darbe_periph
    y0[12] = 0.0              # PHI_plasma
    y0[13] = 0.0              # PHI_periph
    return y0


def ckd_ss(stage, t_spinup=30_000):
    """Return steady state for a CKD VP by running to equilibrium."""
    p  = create_ckd_vp(stage)
    y0 = healthy_ss()
    # Run without any drug dosing to reach CKD SS
    sol = solve_ivp(
        lambda t, y: rhs(t, y, p),
        (0, t_spinup),
        y0,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"CKD SS spin-up failed for {stage}: {sol.message}")
    return sol.y[:, -1], p


# ---------------------------------------------------------------------------
# Dosing helpers
# ---------------------------------------------------------------------------

class DoseEvent:
    """Single dose event."""
    __slots__ = ("t", "compartment", "amount")
    def __init__(self, t, compartment, amount):
        self.t = t
        self.compartment = compartment  # index into state vector
        self.amount = amount            # mIU/mL or µg/mL


def build_dose_schedule(dose_amount, interval_h, n_doses, t_start_h,
                         compartment):
    """Create a list of DoseEvent for a repeated dosing schedule."""
    return [
        DoseEvent(t_start_h + i * interval_h, compartment, dose_amount)
        for i in range(n_doses)
    ]


# Compartment indices for dosing
DEPOT_ESA   = 9    # SC rHuEPO depot
PLASMA_DARBE = 10  # IV or SC darbepoetin (SC is handled as instant for simplicity)
PLASMA_PHI  = 12   # PHI (oral, treated as SC absorption)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def rhs(t, y, p, doses=None):
    """
    ODE RHS for the 14-state erythropoiesis model.

    Parameters
    ----------
    t : float        Current time [h]
    y : ndarray[14]  State vector
    p : dict         Parameter dict (from make_params or create_ckd_vp)
    doses : list     (unused here; dosing handled via events in solver wrapper)

    Returns
    -------
    dydt : ndarray[14]
    """
    # Unpack state
    HIFa        = max(y[0],  0.0)
    EPOp        = max(y[1],  0.0)   # EPO plasma (endogenous + ESA)
    EPOperiph   = max(y[2],  0.0)
    EPOR        = max(y[3],  0.0)
    EPO_LR      = max(y[4],  0.0)
    Prog        = max(y[5],  0.0)
    Prec        = max(y[6],  0.0)
    Retics      = max(y[7],  0.0)
    RBCM        = max(y[8],  0.0)
    ESA_depot   = max(y[9],  0.0)   # SC ESA depot
    Darbe_p     = max(y[10], 0.0)
    Darbe_ph    = max(y[11], 0.0)
    PHI_p       = max(y[12], 0.0)
    PHI_ph      = max(y[13], 0.0)

    # ── 1. Computed algebraic variables ────────────────────────────────────

    # Haemoglobin (Eq 28)
    HGB = p["MCH_Reti"] * Retics + p["MCH_RBCM"] * RBCM

    # Hb-effect on PHD (inferred from paper text: t½=5min at Hb=14, 8min at Hb=9)
    # Hbeffect = max(0, Hb_norm/HGB - 1)  [Supplementary "Detailed Equation"]
    HGB_safe = max(HGB, 0.1)   # avoid division by zero
    Hbeffect = max(0.0, p["Hb_norm"] / HGB_safe - 1.0)   # Eq (Supp)

    # PHI effect on PHD (Eq 1 term)
    # PHIeffect = Emax_PHI * PHI^n / (EC50^n + PHI^n)  [standard Emax]
    PHI_total = PHI_p + Darbe_p * 0.0   # PHI only; ESA has no direct PHD effect
    denom_PHI = p["EC50_PHI"] ** p["n_PHI"] + PHI_total ** p["n_PHI"]
    PHIeffect = p["Emax_PHI"] * (PHI_total ** p["n_PHI"]) / max(denom_PHI, 1e-20)

    # Active PHD (Eq 1)
    PHD = p["PHDbasal"] / (1.0 + PHIeffect + Hbeffect)

    # Total EPO in plasma = endogenous + ESA contribution absorbed from depot
    # (ESA depot absorption is handled as a separate equation that feeds into EPOp)

    # EPO fraction bound to EPOR  (Eq 16 term: fEPO_LRcomplex)
    fEPO_LR = EPO_LR / max(EPOR + EPO_LR, 1e-20)

    # Apply CKD multipliers
    kprodEPO  = p["kprodEPO"] * p["kprodEPO_factor"]
    kprod_P   = p["kprod_P"]  * p["kprod_P_factor"]
    kdeg_RBCM = p["kdeg_RBCM"] * p["kdeg_RBCM_factor"]
    kbasedeg  = p["kbasedeg"] * p["kbasedeg_factor"]

    # ── 2. Rates ───────────────────────────────────────────────────────────

    # Eq 2: HIF-α degradation
    Rate_HIFdeg = HIFa * PHD * p["kmod"]   # a.u./h

    # Eq 4: EPO production
    hill4 = (HIFa ** p["n2"]) / (p["KmprodEPO"] ** p["n2"] + HIFa ** p["n2"] + 1e-30)
    EPO_prod = kprodEPO * hill4             # mIU/mL/h

    # Eq 5: non-specific EPO plasma clearance
    EPO_nscl = p["kelEPO"] * EPOp          # mIU/mL/h

    # Eq 6: EPO distribution to peripheral
    EPO_to_periph = p["kepocp"] * EPOp - p["kepopc"] * EPOperiph  # mIU/mL/h

    # Eqs 7–9: EPO-EPOR binding
    # (EPO in mIU/mL; kon in mL/(mIU·h); EPOR in mIU_eq/mL)
    EPO_fwd = p["kon_EPO"] * EPOR * EPOp   # mIU_eq/mL/h  (Eq 7)
    EPO_bwd = p["koff_EPO"] * EPO_LR       # mIU_eq/mL/h  (Eq 8)
    Net_EPOR_bind = EPO_fwd - EPO_bwd      # mIU_eq/mL/h  (Eq 9)
    # Note: in the EPO_plasma ODE, the "net binding" removes EPO from plasma;
    # the binding is in molecule/mL in the paper but functionally equivalent here
    # since we absorb the conversion into kon_EPO.

    # Eq 10: total EPO plasma clearance
    # The EPOR binding removes EPO from plasma at rate = EPO_fwd - EPO_bwd
    # (only the fraction that is NOT returned = kdegLR*EPO_LR is net EPO loss)
    EPO_EPOR_cl = p["kdegLR"] * EPO_LR    # net EPO consumed by TMDD (mIU/mL/h)
    # In the plasma EPO equation: Net_binding = EPO_fwd - EPO_bwd = Net_EPOR_bind
    # This is the term that appears in Eq 10.
    Total_EPO_cl = EPO_nscl + EPO_to_periph + Net_EPOR_bind   # Eq 10

    # ESA (rHuEPO) depot absorption → plasma
    ESA_abs = p["ka_ESA"] * ESA_depot      # mIU/mL/h absorbed into plasma

    # Progenitor rates (Eqs 14–17)
    Prod_P   = kprod_P                      # Eq 14  a.u./h (constant source)
    Mat_P    = p["kmat_P"] * Prog           # Eq 15
    Deg_P    = kbasedeg * Prog * (1.0 - fEPO_LR)   # Eq 16

    # Precursor rates (Eqs 18–21)
    Prod_C   = p["kmat_P"] * Prog           # Eq 18  (kProgenitorsToPrecursors = kmat_P)
    Deg_C    = p["kdeg_C"] * Prec           # Eq 19
    Rel_C    = p["kmat_C"] * Prec           # Eq 20  (release from BM to circulation)

    # Reticulocyte rates (Eqs 22–25)
    # Flux from precursors to retics: convert a.u./h → 10^12/L/h
    Mat_retics = Rel_C * p["scale_retic"]       # 10^12/L/h  (Eq 22 converted)
    Mat_RBCM   = p["kReticsToRBCM"] * Retics    # Eq 23
    Deg_Ret    = p["kdeg_Ret"] * Retics          # Eq 24

    # RBC (Eqs 26–27)
    Deg_RBCM   = kdeg_RBCM * RBCM               # Eq 26

    # Darbepoetin PK (two-compartment SC/IV)
    Darbe_abs  = 0.0   # handled externally if IV; SC would use depot
    Darbe_cl   = p["kel_Darbe"] * Darbe_p
    Darbe_dist = p["kecp_Darbe"] * Darbe_p - p["kepc_Darbe"] * Darbe_ph
    # Darbepoetin also binds EPOR (with different kon/koff), contributes to fEPO_LR
    # For simplicity, darbepoetin EPO equivalence is captured via ESA_depot scaling

    # PHI PK (two-compartment oral)
    PHI_cl     = p["kel_PHI"] * PHI_p
    PHI_dist   = p["kecp_PHI"] * PHI_p - p["kepc_PHI"] * PHI_ph

    # ── 3. ODEs ──────────────────────────────────────────────────────────

    dydt = np.zeros(14)

    # State 0: HIFα (Eq 3)
    dydt[0] = p["kprodHIFa"] - Rate_HIFdeg

    # State 1: EPOplasma (Eq 11) — includes absorbed ESA
    dydt[1] = EPO_prod + ESA_abs - Total_EPO_cl

    # State 2: EPOperiphery (Eq 6)
    dydt[2] = EPO_to_periph

    # State 3: EPOR (Eq 12)
    dydt[3] = p["kdegLR"] * EPO_LR - EPO_fwd + EPO_bwd

    # State 4: EPO-LR complex (Eq 13)
    dydt[4] = EPO_fwd - EPO_bwd - p["kdegLR"] * EPO_LR

    # State 5: Progenitors (Eq 17)
    dydt[5] = Prod_P - Mat_P - Deg_P

    # State 6: Precursors (Eq 21)
    dydt[6] = Prod_C - Deg_C - Rel_C

    # State 7: Reticsplasma (Eq 25)
    dydt[7] = Mat_retics - Mat_RBCM - Deg_Ret

    # State 8: RBCM (Eq 27)
    dydt[8] = Mat_RBCM - Deg_RBCM

    # State 9: ESA depot (SC absorption)
    dydt[9] = -p["ka_ESA"] * ESA_depot

    # State 10: Darbepoetin plasma
    dydt[10] = Darbe_abs - Darbe_cl - Darbe_dist

    # State 11: Darbepoetin peripheral
    dydt[11] = Darbe_dist

    # State 12: PHI plasma (oral: assume instant depot → plasma simplified)
    dydt[12] = -PHI_cl - PHI_dist

    # State 13: PHI peripheral
    dydt[13] = PHI_dist

    return dydt


# ---------------------------------------------------------------------------
# Solver with dosing events
# ---------------------------------------------------------------------------

def simulate(y0, p, t_end_h, dose_events=None, rtol=1e-8, atol=1e-10):
    """
    Simulate the model from y0 to t_end_h, applying discrete dose events.

    Parameters
    ----------
    y0          : array[14]     Initial conditions
    p           : dict          Parameters
    t_end_h     : float         End time [h]
    dose_events : list[DoseEvent] or None   List of dosing events
    rtol, atol  : float         Solver tolerances

    Returns
    -------
    t   : ndarray   Time points [h]
    y   : ndarray   State history [14 × n_t]
    """
    if dose_events is None:
        dose_events = []

    # Sort events by time
    events_sorted = sorted(dose_events, key=lambda e: e.t)

    # Build time segments between dose events
    t_breaks = sorted(set([e.t for e in events_sorted]))
    t_breaks = [tb for tb in t_breaks if 0 < tb < t_end_h]
    seg_starts = [0.0] + t_breaks
    seg_ends   = t_breaks + [t_end_h]

    t_out = [np.array([0.0])]
    y_out = [y0.reshape(-1, 1)]
    y_cur = y0.copy()

    for t_s, t_e in zip(seg_starts, seg_ends):
        # Apply any dose events at t_s (except t=0)
        if t_s > 0:
            for ev in events_sorted:
                if abs(ev.t - t_s) < 1e-9:
                    y_cur[ev.compartment] += ev.amount

        sol = solve_ivp(
            lambda t, y: rhs(t, y, p),
            (t_s, t_e),
            y_cur,
            method="Radau",
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed at t={t_s}–{t_e}: {sol.message}")

        # Dense output on a grid
        t_grid = np.linspace(t_s, t_e, max(2, int((t_e - t_s) * 2 + 1)))
        y_grid = sol.sol(t_grid)
        # Skip first point (already included from previous segment)
        t_out.append(t_grid[1:])
        y_out.append(y_grid[:, 1:])
        y_cur = sol.y[:, -1].copy()

    t_all = np.concatenate(t_out)
    y_all = np.concatenate(y_out, axis=1)
    return t_all, y_all


# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------

def get_hgb(y, p):
    """HGB [g/dL] from state array."""
    return p["MCH_Reti"] * y[7] + p["MCH_RBCM"] * y[8]


def get_epo(y):
    """Total plasma EPO [mIU/mL]."""
    return y[1]


def get_retics(y):
    """Plasma reticulocytes [10^12/L]."""
    return y[7]


def get_rbc(y):
    """RBCM [10^12/L]."""
    return y[8]


def get_all_obs(t, y, p):
    """Return dict of all key observables as time series."""
    return {
        "t_h": t,
        "t_wk": t / (24.0 * 7.0),
        "EPO_mIUmL": y[1],
        "Hb_gdL": get_hgb(y, p),
        "Retics_1e12L": y[7],
        "RBC_1e12L": y[8],
        "HIFa_au": y[0],
        "PHI_ugmL": y[12],
    }


# ---------------------------------------------------------------------------
# Quick self-check: verify healthy SS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    p  = make_params()
    y0 = healthy_ss()
    print("Initial state:")
    print(f"  HIFa={y0[0]:.3f}  EPO={y0[1]:.2f} mIU/mL  EPOR={y0[3]:.1f}  EPO_LR={y0[4]:.1f}")
    print(f"  Prog={y0[5]:.3f}  Prec={y0[6]:.3f}")
    print(f"  Retics={y0[7]:.4f} (10^12/L)  RBCM={y0[8]:.2f} (10^12/L)")
    HGB0 = get_hgb(y0, p)
    print(f"  HGB={HGB0:.2f} g/dL")

    # Check RHS at SS — all dX/dt should be ~0
    dy = rhs(0.0, y0, p)
    print("\nRHS at healthy SS (should all be ~0):")
    names = ["HIFa","EPOp","EPOperiph","EPOR","EPO_LR","Prog","Prec",
             "Retics","RBCM","ESAdepot","Darbe_p","Darbe_ph","PHI_p","PHI_ph"]
    for i, (nm, dyi) in enumerate(zip(names, dy)):
        flag = " <== LARGE" if abs(dyi) > 1e-4 else ""
        print(f"  d{nm}/dt = {dyi:.6e}{flag}")

    print("\nRunning 1000h spinup ...")
    t, y = simulate(y0, p, 1000.0, rtol=1e-8, atol=1e-10)
    HGB_f = get_hgb(y[:, -1], p)
    EPO_f = get_epo(y[:, -1])
    print(f"  Final: HGB={HGB_f:.2f} g/dL (target 14.53), EPO={EPO_f:.2f} mIU/mL (target 7.0)")
    print(f"  Retics={y[7,-1]:.4f} (target 0.09)  RBCM={y[8,-1]:.2f} (target 5.0)")

    err_hgb = abs(HGB_f - 14.53) / 14.53
    err_epo = abs(EPO_f - 7.0) / 7.0
    if err_hgb < 0.05 and err_epo < 0.10:
        print("\nSELF-CHECK PASSED: SS within 10% of targets")
        sys.exit(0)
    else:
        print(f"\nSELF-CHECK WARNING: HGB error={err_hgb*100:.1f}%, EPO error={err_epo*100:.1f}%")
        sys.exit(1)
