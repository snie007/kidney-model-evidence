"""
model.py — Peng 2001 whole-body PBPK model for ASO ISIS 1082 in rat.

Reference:
  Peng B, Andrews J, Nestorov I, Brennan B, Nicklin P, Rowland M.
  "Tissue Distribution and Physiologically Based Pharmacokinetics of Antisense
  Phosphorothioate Oligonucleotide ISIS 1082 in Rat."
  Antisense Nucleic Acid Drug Dev. 2001;11(1):15-27.
  Local PDF: resources/papers/P2-ASO-001_Peng_2001_ASO_ISIS1082_PBPK_rat.pdf

Model structure:
  Two-compartment permeability-limited model for each organ.
  Per tissue: vascular compartment (CT_V) + extravascular compartment (CT_EV).
  Closed circulatory system: venous → lungs → arterial → tissues → venous.

Equations (Appendix 1):
  General tissue (vascular):
    VT_V * dCT_V/dt = -[QT + fub*(PST + CLuT_V)]*CT_V + (PST/KpuT)*CT_EV + InpT
  General tissue (extravascular):
    VT_EV * dCT_EV/dt = fub*PST*CT_V - (PST + CLuT_EV)/KpuT * CT_EV
  Arterial blood:
    VART * dCART/dt = -QART*CART + QART*CLU_V   (QART = total cardiac output)
  Venous blood:
    VVEN * dCVEN/dt = -QVEN*CVEN + sum(QT*CT_V) for all tissues except lung
    CVEN(0) = dose/VVEN  (iv bolus)

Parameters from Table 1 (250 g rat) and Table 2 (fitted):
  - Kp = tissue/plasma partition coefficient (Table 2)
  - fub*PS = effective permeability surface area product mL/h (Table 2)
  - CLuR = renal unbound clearance = 2.11 mL/h (Table 2)
  - fu = 0.105, fub = 0.105 (R = blood/plasma ratio = 1)
  - Kpu = Kp/fub (unbound partition coefficient)

Units:
  Concentrations: µg/g (or µg/mL assuming density 1)
  Time: hours
  Volumes: mL (using tissue weights in grams ≈ mL)
  Blood flows: mL/h
  Dose: µg (10 mg/kg * 250 g rat = 2500 µg)

Tissues (abbreviations):
  ART = arterial blood, VEN = venous blood, LU = lung, LI = liver,
  KI = kidney, ST = stomach, SP = spleen, IN = intestine, MU = muscle,
  AD = adipose, SK = skin, BO = bone, HT = heart, TH = thymus, BR = brain, TE = testes
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── PHYSIOLOGICAL PARAMETERS (Table 1, 250 g rat) ────────────────────────────
# Blood flows (mL/h)
Q = {
    "LU": 2670,   # cardiac output = total lung flow
    "LI": 162,    "KI": 684,    "ST":  66,    "SP":  37.8,
    "IN": 450,    "MU": 408,    "AD": 114,    "SK":  264,
    "BO": 150,    "HT": 210,    "TH":  19.2,  "BR":  78,
    "TE":  27,
}
# Portal vein inputs to liver
Q["HE"] = Q["LI"]  # hepatic artery flow
# QST, QSP, QIN drain into liver portal; Q["LI"] = Q["HE"]+Q["ST"]+Q["SP"]+Q["IN"]
Q["LI"] = Q["HE"] + Q["ST"] + Q["SP"] + Q["IN"]  # total liver flow = 162+66+37.8+450 = 715.8
# Total cardiac output = sum of all peripheral tissue flows (excluding venous→lung→art)
# Note: paper Table 1 gives QKI=684 mL/h but uses QART=QVEN=2670 mL/h (total CO)

# Vascular volumes (mL ≈ g, Table 1)
V_V = {
    "ART": 5.70, "VEN": 11.3,
    "LU":  0.42, "LI":  1.27, "KI":  0.24, "ST":  0.15, "SP":  0.37,
    "IN":  1.30, "MU":  3.21, "AD":  1.45, "SK":  0.76, "BO":  0.65,
    "HT":  0.15, "TH":  0.02, "BR":  0.06, "TE":  0.02,
}
# Extravascular volumes (mL)
V_EV = {
    "LU":  1.18, "LI":  9.74, "KI":  2.06, "ST":  0.96, "SP":  0.93,
    "IN":  9.70, "MU": 119.0, "AD":  8.55, "SK": 39.2,  "BO": 15.2,
    "HT":  1.13, "TH":  0.68, "BR":  1.64, "TE":  2.48,
}

# ── DRUG-SPECIFIC PARAMETERS (Table 2) ───────────────────────────────────────
fu  = 0.105   # unbound fraction in plasma
fub = 0.105   # unbound fraction in whole blood (= fu; blood/plasma ratio = 1)

# Tissue/plasma partition coefficients (Kp, dimensionless)
Kp = {
    "LU":  2.77, "LI": 12.7,  "KI": 38.8,  "ST":  1.41, "SP":  6.04,
    "IN":  2.40, "MU":  0.94, "AD":  0.66,  "SK":  0.97, "BO":  2.63,
    "HT":  1.28, "TH":  0.86, "BR":  0.89,  "TE":  1.68,
}
Kpu = {t: Kp[t] / fub for t in Kp}  # unbound partition coefficient

# Effective permeability surface area products (fub*PS, mL/h)
fubPS = {
    "LU":  0.26, "LI": 10.1,  "KI":  4.69, "ST":  0.25, "SP":  0.49,
    "IN":  3.34, "MU":  7.16, "AD":  0.67,  "SK":  9.15, "BO": 17.1,
    "HT":  0.31, "TH":  1.37, "BR":  0.03,  "TE":  0.18,
}
# Derive PS from fub*PS: PS = fubPS / fub
PS = {t: fubPS[t] / fub for t in fubPS}

# Renal clearance (filtration-based for ISIS 1082)
CLuR = 2.11  # mL/h (Table 2)

# ── TISSUE ORDER (for ODE vector) ────────────────────────────────────────────
TISSUES = ["LU", "LI", "KI", "ST", "SP", "IN", "MU", "AD", "SK", "BO",
           "HT", "TH", "BR", "TE"]
# Tissues that drain to venous (not liver-draining splanchnic)
VENOUS_DRAINING = ["KI", "LI", "MU", "AD", "SK", "BO", "HT", "BR", "TE", "TH"]
# Liver receives arterial + portal (ST, SP, IN)
PORTAL_TISSUES = ["ST", "SP", "IN"]


def _build_index():
    """Return dict: state name → ODE index."""
    idx = {}
    i = 0
    idx["C_ART"] = i; i += 1
    idx["C_VEN"] = i; i += 1
    for t in TISSUES:
        idx[f"C_{t}_V"] = i;  i += 1
        idx[f"C_{t}_EV"] = i; i += 1
    return idx


IDX = _build_index()
N_STATES = len(IDX)


def get_params(dose_mg_per_kg=10.0, bw_g=250.0):
    """Return parameter dict for the model."""
    dose_ug = dose_mg_per_kg * bw_g  # µg (10 mg/kg * 250 g = 2500 µg)
    return {
        "dose_ug": dose_ug,
        "Q": Q, "V_V": V_V, "V_EV": V_EV,
        "Kp": Kp, "Kpu": Kpu, "PS": PS, "fubPS": fubPS,
        "fu": fu, "fub": fub,
        "CLuR": CLuR,
    }


def rhs(t, y, p):
    """ODE right-hand side."""
    dv = np.zeros(N_STATES)

    C_ART = y[IDX["C_ART"]]
    C_VEN = y[IDX["C_VEN"]]

    # Convenience extractors
    def C_V(tissue):
        return y[IDX[f"C_{tissue}_V"]]
    def C_EV(tissue):
        return y[IDX[f"C_{tissue}_EV"]]

    # ── Arterial blood ──────────────────────────────────────────────────────
    Q_CO = p["Q"]["LU"]   # cardiac output = lung flow
    C_LU_V = C_V("LU")
    dv[IDX["C_ART"]] = (Q_CO * C_LU_V - Q_CO * C_ART) / p["V_V"]["ART"]

    # ── Venous blood ────────────────────────────────────────────────────────
    Q_ven = p["Q"]["LU"]  # all venous flow goes to lung
    inflow_ven = sum(p["Q"][tx] * y[IDX[f"C_{tx}_V"]] for tx in VENOUS_DRAINING)
    dv[IDX["C_VEN"]] = (inflow_ven - Q_ven * C_VEN) / p["V_V"]["VEN"]

    # ── Lung ────────────────────────────────────────────────────────────────
    tissue = "LU"
    Qt   = p["Q"][tissue]
    Vv   = p["V_V"][tissue]
    Vev  = p["V_EV"][tissue]
    PSt  = p["PS"][tissue]
    kpu  = p["Kpu"][tissue]
    Cv   = C_V(tissue)
    Cev  = C_EV(tissue)
    Inp  = Q_ven * C_VEN        # lungs receive venous blood
    dv[IDX[f"C_{tissue}_V"]]  = (-Qt*Cv - p["fub"]*PSt*Cv + PSt/kpu*Cev + Inp) / Vv
    dv[IDX[f"C_{tissue}_EV"]] = (p["fub"]*PSt*Cv - PSt/kpu*Cev) / Vev

    # ── Liver (portal drainage from ST, SP, IN) ─────────────────────────────
    tissue = "LI"
    Qt   = p["Q"][tissue]      # total liver inflow
    QHE  = Qt - p["Q"]["ST"] - p["Q"]["SP"] - p["Q"]["IN"]   # hepatic artery
    Vv   = p["V_V"][tissue]
    Vev  = p["V_EV"][tissue]
    PSt  = p["PS"][tissue]
    kpu  = p["Kpu"][tissue]
    Cv   = C_V(tissue)
    Cev  = C_EV(tissue)
    Inp  = (QHE * C_ART
            + p["Q"]["ST"] * C_V("ST")
            + p["Q"]["SP"] * C_V("SP")
            + p["Q"]["IN"] * C_V("IN"))
    dv[IDX[f"C_{tissue}_V"]]  = (-Qt*Cv - p["fub"]*PSt*Cv + PSt/kpu*Cev + Inp) / Vv
    dv[IDX[f"C_{tissue}_EV"]] = (p["fub"]*PSt*Cv - PSt/kpu*Cev) / Vev

    # ── Kidney (has renal clearance from vascular) ──────────────────────────
    tissue = "KI"
    Qt   = p["Q"][tissue]
    Vv   = p["V_V"][tissue]
    Vev  = p["V_EV"][tissue]
    PSt  = p["PS"][tissue]
    kpu  = p["Kpu"][tissue]
    CLuT_V = p["CLuR"]         # renal unbound vascular clearance
    Cv   = C_V(tissue)
    Cev  = C_EV(tissue)
    Inp  = Qt * C_ART
    dv[IDX[f"C_{tissue}_V"]]  = (-Qt*Cv - p["fub"]*(PSt + CLuT_V)*Cv
                                  + PSt/kpu*Cev + Inp) / Vv
    dv[IDX[f"C_{tissue}_EV"]] = (p["fub"]*PSt*Cv - PSt/kpu*Cev) / Vev

    # ── All other non-eliminating tissues ──────────────────────────────────
    for tissue in TISSUES:
        if tissue in ("LU", "LI", "KI"):
            continue
        Qt   = p["Q"][tissue]
        Vv   = p["V_V"][tissue]
        Vev  = p["V_EV"][tissue]
        PSt  = p["PS"][tissue]
        kpu  = p["Kpu"][tissue]
        Cv   = C_V(tissue)
        Cev  = C_EV(tissue)
        Inp  = Qt * C_ART
        dv[IDX[f"C_{tissue}_V"]]  = (-Qt*Cv - p["fub"]*PSt*Cv
                                      + PSt/kpu*Cev + Inp) / Vv
        dv[IDX[f"C_{tissue}_EV"]] = (p["fub"]*PSt*Cv - PSt/kpu*Cev) / Vev

    return dv


def simulate(dose_mg_per_kg=10.0, bw_g=250.0, t_end_h=72.0,
             t_eval=None, rtol=1e-8, atol=1e-10):
    """
    Run the Peng 2001 whole-body PBPK simulation.

    Returns:
        sol : scipy OdeSolution object
        p   : parameter dict
    """
    p = get_params(dose_mg_per_kg, bw_g)

    y0 = np.zeros(N_STATES)
    # IV bolus: dose deposited in venous blood at t=0
    y0[IDX["C_VEN"]] = p["dose_ug"] / p["V_V"]["VEN"]

    if t_eval is None:
        t_eval = np.concatenate([
            np.linspace(0, 1, 20),
            np.linspace(1, 24, 60),
            np.linspace(24, t_end_h, 20),
        ])
        t_eval = np.unique(t_eval)

    sol = solve_ivp(
        rhs, [0, t_end_h], y0, method="LSODA", args=(p,),
        t_eval=t_eval, rtol=rtol, atol=atol, dense_output=True,
    )
    return sol, p


def get_tissue_conc(sol, p, tissue, compartment="total"):
    """
    Extract tissue concentration vs time from ODE solution.

    compartment: 'vascular', 'extravascular', or 'total'
    Returns: (t, C) arrays, concentrations in µg/g
    """
    t = sol.t
    Cv  = sol.y[IDX[f"C_{tissue}_V"]]
    Cev = sol.y[IDX[f"C_{tissue}_EV"]]
    Vv  = p["V_V"][tissue]
    Vev = p["V_EV"][tissue]
    Vtot = Vv + Vev

    if compartment == "vascular":
        return t, Cv
    elif compartment == "extravascular":
        return t, Cev
    else:  # total: mass-weighted average
        C_tot = (Cv * Vv + Cev * Vev) / Vtot
        return t, C_tot


def get_arterial_conc(sol):
    """Return (t, C_arterial) arrays."""
    return sol.t, sol.y[IDX["C_ART"]]
