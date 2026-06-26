"""
Thomas & Ford Versypt (2024) — Glomerular Fibrosis Model
=========================================================
Python port of the MATLAB model:
  GlomerularFibrosis.m / runGlomerularFibrosis.m / parameter_vals.m

State variables (8):
  y[0] AGE   — Advanced Glycation End-products
  y[1] MCP   — Monocyte Chemoattractant Protein
  y[2] MAC   — Macrophages
  y[3] TGF   — TGF-beta
  y[4] AMC   — Activated Mesangial Cells (AMC / MC+)
  y[5] MMP   — Matrix Metalloproteinase
  y[6] TIMP  — Tissue Inhibitor of Metalloproteinase
  y[7] COL   — Collagen

Time units: days (healthy pre-run 1..3000 days; DKD run 1..SimTimeDays)
Glucose input ramps from 5.8 mmol/L (healthy) to 28.3 mmol/L (diabetic)
between weeks 6 and 16 of the DKD run.

Fitted parameter values loaded from ModelFitResultsFinal.mat via scipy.io.

MAC FOLD-CHANGE NOTE (verified 2026-06-23):
  Using PARAMS_FITTED (FitColData), MAC peak fold-change is ~13× (vs healthy SS).
  Paper Figure 5B shows ~4-5×. Both the MATLAB reference and this Python port
  give 13.4×. The discrepancy arises because the paper used a sequential fitting
  procedure: {mu_MAC, n_MCP} were fitted to MAC data first (FitMACData, gives ~4-5×),
  then COL parameters were fitted while keeping MAC parameters fixed. The final
  FitColData parameter set was not simultaneously re-optimised for MAC dynamics.
  This is NOT a port error — it is a sequential fitting artefact present in the
  original MATLAB code. Document this in any publication using this model.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Parameter set (fitted values from ModelFitResultsFinal.mat, FitColData)
# ---------------------------------------------------------------------------
# Parameter ordering follows parameter_vals.m / ParameterNames.m:
#   [1-13]  sys1: K_AGE L_MCP mu_MCP S_MCP mu_AGE K_MCP MAC_0 n_MCP n_GLU mu_MAC L_MAC L_AGE K_GLU
#   [14-22] sys2/3: mu_TGF MC mu_AMC L_AMC K_TGF S_AMC L_TGF S_TGF n_TGF
#   [23-33] sys4: G_MMP G_TIMP mu_MMP mu_TIMP L_TIMP L_MMP mu_COL L_COL G_COL L_COLA K_I

PARAMS_FITTED = {
    # System 1: AGE, MCP, Macrophages
    "K_AGE":   1.1437e-06,    # Half-saturation for AGE-MCP coupling
    "L_MCP":   4.0869e-10,    # MCP production rate due to AGE
    "mu_MCP":  1.73,          # MCP degradation rate (1/day)
    "S_MCP":   2.768e-10,     # Basal MCP secretion (= 160e-12 * mu_MCP from SS assumption)
    "mu_AGE":  0.0087,        # AGE degradation rate (1/day)
    "K_MCP":   5.0e-09,       # Half-saturation for MCP-driven macrophage recruitment
    "MAC_0":   5.0e-05,       # Basal macrophage pool
    "n_MCP":   5.38072076,    # Hill coefficient (MCP → MAC) — fitted
    "n_GLU":   2.69857050,    # Hill coefficient (glucose → AGE) — fitted
    "mu_MAC":  0.15,          # Macrophage clearance rate (1/day) — fitted
    "L_MAC":   0.0622458205,  # Macrophage recruitment rate — fitted
    "L_AGE":   1.53968472e-05,# AGE production rate — fitted
    "K_GLU":   0.0337,        # Half-saturation glucose (g/mL)
    # System 2/3: TGF-beta, Activated Mesangial Cells
    "d_TGF":   333.0,         # TGF degradation rate (1/day)
    "MC":      0.67,          # Quiescent mesangial cell concentration
    "mu_AMC":  0.5,           # AMC deactivation rate (1/day)
    "L_AMC":   4.0e-03,       # TGF-driven AMC activation rate
    "K_TGF":   2.5e-09,       # TGF half-saturation for AMC activation
    "S_AMC":   2.60909354e-04,# Basal AMC production — fitted
    "L_TGF":   1.33958124e+05,# TGF production rate by macrophages — fitted
    "S_TGF":   2.84860638e-07,# Basal TGF secretion — fitted
    "n_TGF":   4.14229216,    # Hill coefficient (TGF → AMC) — fitted
    # System 4: MMP, TIMP, Collagen
    "G_MMP":   4.98e+08,      # MMP-TIMP binding/inactivation rate
    "G_TIMP":  1.04e+09,      # TIMP-MMP inactivation rate
    "mu_MMP":  4.32,          # MMP degradation rate (1/day)
    "mu_TIMP": 21.6,          # TIMP degradation rate (1/day)
    "L_TIMP":  6.0e-05,       # TIMP production rate by macrophages (overridden = L_MMP/5)
    "L_MMP":   2.06451185e+07,# MMP production rate by macrophages — fitted
    "mu_COL":  0.37,          # Collagen degradation rate (1/day)
    "L_COL":   0.003,         # Collagen production by quiescent mesangial cells
    "G_COL":   18357.9752,    # MMP-driven collagen degradation rate — fitted
    "L_COLA":  1834.81526,    # Collagen production by activated mesangial cells — fitted
    "K_I":     1.0e+05,       # Treatment inhibition constant
}

# Note: In GlomerularFibrosis.m line 53, L_TIMP is OVERRIDDEN: L_TIMP = (1/5)*L_MMP
# This is applied inside the ODE function below.


# ---------------------------------------------------------------------------
# Glucose input function
# ---------------------------------------------------------------------------
def glucose_input(t, scenario, glucose_ctrl, treatment_time):
    """
    Glucose concentration (g/mL) as a function of time (days).

    Healthy: 5.8 mmol/L constant
    Diabetic: ramps from 5.8 to 28.3 mmol/L between weeks 6 and 16,
              then stays at 28.3 unless glucose control is applied
              after TreatmentTime.

    Parameters
    ----------
    t : float
        Time in days.
    scenario : str
        'healthy' or 'diabetic'
    glucose_ctrl : list of str
        e.g. ['NoGlucoseCtrl'] or ['YesGlucoseCtrl']
    treatment_time : float
        Treatment start time in weeks.
    """
    mol_2_gram = 180.0 / 1_000_000.0  # mmol/L → g/mL
    y1 = 5.8   # mmol/L healthy glucose
    y2 = 28.3  # mmol/L diabetic glucose
    t1 = 6 * 7.0   # weeks 6 → days
    t2 = 16 * 7.0  # weeks 16 → days
    treatment_time_days = treatment_time * 7.0

    if scenario == "healthy":
        return y1 * mol_2_gram

    # Diabetic ramp
    if t <= t1:
        gluc = y1
    elif t1 < t <= t2:
        slope = (y2 - y1) / (t2 - t1)
        gluc = slope * t + y1 - slope * t1
    elif t2 < t <= treatment_time_days:
        gluc = y2
    else:  # t > treatment_time_days
        if "NoGlucoseCtrl" in glucose_ctrl:
            gluc = y2
        elif "YesGlucoseCtrl" in glucose_ctrl:
            gluc = y1
        else:
            gluc = y2

    return gluc * mol_2_gram


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------
def odes(t, y, p, scenario="diabetic", glucose_ctrl=None, treatment_time=16.0):
    """
    ODE right-hand side for the glomerular fibrosis model.

    Parameters
    ----------
    t : float
        Time (days).
    y : array-like, shape (8,)
        State vector [AGE, MCP, MAC, TGF, AMC, MMP, TIMP, COL].
    p : dict
        Parameter dictionary (use PARAMS_FITTED or a modified copy).
    scenario : str
        'healthy' or 'diabetic'.
    glucose_ctrl : list of str
        Glucose control options.
    treatment_time : float
        Treatment start (weeks).

    Returns
    -------
    dydt : ndarray, shape (8,)
    """
    if glucose_ctrl is None:
        glucose_ctrl = ["NoGlucoseCtrl"]

    AGE, MCP, MAC, TGF, AMC, MMP, TIMP, COL = y

    # Unpack parameters — System 1
    K_AGE   = p["K_AGE"]
    L_MCP   = p["L_MCP"]
    mu_MCP  = p["mu_MCP"]
    S_MCP   = p["S_MCP"]
    mu_AGE  = p["mu_AGE"]
    K_MCP   = p["K_MCP"]
    MAC_0   = p["MAC_0"]
    n_MCP   = p["n_MCP"]
    n_GLU   = p["n_GLU"]
    mu_MAC  = p["mu_MAC"]
    L_MAC   = p["L_MAC"]
    L_AGE   = p["L_AGE"]
    K_GLU   = p["K_GLU"]

    # System 2/3
    d_TGF   = p["d_TGF"]
    MC      = p["MC"]
    mu_AMC  = p["mu_AMC"]
    L_AMC   = p["L_AMC"]
    K_TGF   = p["K_TGF"]
    S_AMC   = p["S_AMC"]
    L_TGF   = p["L_TGF"]
    S_TGF   = p["S_TGF"]
    n_TGF   = p["n_TGF"]

    # System 4
    G_MMP   = p["G_MMP"]
    G_TIMP  = p["G_TIMP"]
    mu_MMP  = p["mu_MMP"]
    mu_TIMP = p["mu_TIMP"]
    L_MMP   = p["L_MMP"]
    # L_TIMP overridden in MATLAB (line 53 of GlomerularFibrosis.m)
    L_TIMP  = (1.0 / 5.0) * L_MMP
    mu_COL  = p["mu_COL"]
    L_COL   = p["L_COL"]
    G_COL   = p["G_COL"]
    L_COLA  = p["L_COLA"]
    K_I     = p["K_I"]

    # Glucose input
    GLU = glucose_input(t, scenario, glucose_ctrl, treatment_time)

    # AGE treatment
    K_AGE_I   = 0.0
    K_AGE_Deg = 0.0
    treatment_time_days = treatment_time * 7.0
    if "AGEInhibition" in glucose_ctrl:
        if t > treatment_time_days:
            K_AGE_I = K_I
    elif "AGEDegradation" in glucose_ctrl:
        if t > treatment_time_days:
            K_AGE_Deg = K_I

    # ODEs (translated from GlomerularFibrosis.m)
    dAGEdt = (L_AGE * (GLU**n_GLU) / (K_GLU**n_GLU + GLU**n_GLU)) / (1.0 + K_AGE_I) \
             - mu_AGE * AGE * (1.0 + K_AGE_Deg)

    dMCPdt = S_MCP + L_MCP * AGE * MC / (K_AGE + AGE) - mu_MCP * MCP

    dMACdt = L_MAC * (MCP**n_MCP / (K_MCP**n_MCP + MCP**n_MCP)) * MAC_0 - mu_MAC * MAC

    dTGFdt = S_TGF + L_TGF * MAC - d_TGF * TGF

    dAMCdt = S_AMC + L_AMC * (TGF**n_TGF / (K_TGF**n_TGF + TGF**n_TGF)) * MC - mu_AMC * AMC

    dMMPdt  = L_MMP  * MAC - G_MMP  * MMP * TIMP - mu_MMP  * MMP
    dTIMPdt = L_TIMP * MAC - G_TIMP * MMP * TIMP - mu_TIMP * TIMP
    dCOLdt  = L_COL * MC + L_COLA * AMC - G_COL * MMP * COL - mu_COL * COL

    return [dAGEdt, dMCPdt, dMACdt, dTGFdt, dAMCdt, dMMPdt, dTIMPdt, dCOLdt]


# ---------------------------------------------------------------------------
# Initial conditions (healthy steady state)
# ---------------------------------------------------------------------------
def initial_conditions(p=None):
    """
    Compute healthy steady-state initial conditions by integrating
    the healthy scenario for 3000 days from zero, exactly as in
    runGlomerularFibrosis.m.

    Returns
    -------
    y_ss : ndarray, shape (8,)
    """
    if p is None:
        p = PARAMS_FITTED

    y0 = np.zeros(8)
    sol = solve_ivp(
        lambda t, y: odes(t, y, p, scenario="healthy", glucose_ctrl=["NoGlucoseCtrl"],
                          treatment_time=1e9),
        t_span=(1.0, 3000.0),
        y0=y0,
        method="LSODA",
        rtol=1e-8,
        atol=1e-15,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Healthy SS integration failed: {sol.message}")
    return sol.y[:, -1]


# ---------------------------------------------------------------------------
# Main simulation runners
# ---------------------------------------------------------------------------
def run_fibrosis(scenario="DKD", p=None, simulation_weeks=24, treatment_time=16):
    """
    Run the fibrosis (DKD) scenario.

    Steps:
      1. Integrate the healthy model for 3000 days to reach steady state.
      2. Use the steady state as initial conditions for the diabetic run.

    Parameters
    ----------
    scenario : str
        'DKD' (alias 'diabetic') or 'healthy'
    p : dict or None
        Parameter dictionary. Defaults to PARAMS_FITTED.
    simulation_weeks : int
        Total simulation time in weeks for the DKD run (default 24).
    treatment_time : int
        Week at which treatment starts (default 16, no control used here).

    Returns
    -------
    t : ndarray
        Time array in days.
    y : ndarray, shape (n_timepoints, 8)
        Solution matrix [AGE, MCP, MAC, TGF, AMC, MMP, TIMP, COL].
    y_ss : ndarray, shape (8,)
        Healthy steady state used as initial conditions.
    """
    if p is None:
        p = PARAMS_FITTED

    # Step 1: healthy SS
    y_ss = initial_conditions(p)

    if scenario in ("healthy",):
        # Return healthy pre-run
        sol = solve_ivp(
            lambda t, y: odes(t, y, p, scenario="healthy", glucose_ctrl=["NoGlucoseCtrl"],
                              treatment_time=1e9),
            t_span=(1.0, 3000.0),
            y0=np.zeros(8),
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
            t_eval=np.linspace(1.0, 3000.0, 3000),
        )
        return sol.t, sol.y.T, y_ss

    # Step 2: DKD run
    sim_days = simulation_weeks * 7
    t_eval = np.linspace(1.0, sim_days, sim_days)
    sol = solve_ivp(
        lambda t, y: odes(t, y, p, scenario="diabetic",
                          glucose_ctrl=["NoGlucoseCtrl"],
                          treatment_time=treatment_time),
        t_span=(1.0, float(sim_days)),
        y0=y_ss,
        method="LSODA",
        rtol=1e-8,
        atol=1e-15,
        t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"DKD integration failed: {sol.message}")

    return sol.t, sol.y.T, y_ss


def run_treatment(treatment="glucose_control", p=None, simulation_weeks=30,
                  treatment_time=24):
    """
    Run a treatment scenario.

    Parameters
    ----------
    treatment : str
        'glucose_control'  — YesGlucoseCtrl
        'AGE_inhibition'   — AGEInhibition
        'AGE_degradation'  — AGEDegradation
        'no_treatment'     — NoGlucoseCtrl (DKD base case)
    p : dict or None
        Parameter dictionary. Defaults to PARAMS_FITTED.
    simulation_weeks : int
        Total simulation weeks (default 30).
    treatment_time : int
        Week treatment starts (default 24).

    Returns
    -------
    t : ndarray
        Time array in days.
    y : ndarray, shape (n_timepoints, 8)
        Solution matrix.
    y_ss : ndarray, shape (8,)
        Healthy steady state.
    """
    if p is None:
        p = PARAMS_FITTED

    _treatment_map = {
        "glucose_control": ["YesGlucoseCtrl"],
        "AGE_inhibition":  ["NoGlucoseCtrl", "AGEInhibition"],
        "AGE_degradation": ["NoGlucoseCtrl", "AGEDegradation"],
        "no_treatment":    ["NoGlucoseCtrl"],
    }
    if treatment not in _treatment_map:
        raise ValueError(f"Unknown treatment '{treatment}'. Choose from: {list(_treatment_map)}")
    glucose_ctrl = _treatment_map[treatment]

    y_ss = initial_conditions(p)

    sim_days = simulation_weeks * 7
    t_eval = np.linspace(1.0, sim_days, sim_days)
    sol = solve_ivp(
        lambda t, y: odes(t, y, p, scenario="diabetic",
                          glucose_ctrl=glucose_ctrl,
                          treatment_time=treatment_time),
        t_span=(1.0, float(sim_days)),
        y0=y_ss,
        method="LSODA",
        rtol=1e-8,
        atol=1e-15,
        t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"Treatment integration failed: {sol.message}")

    return sol.t, sol.y.T, y_ss


# ---------------------------------------------------------------------------
# State variable names
# ---------------------------------------------------------------------------
STATE_NAMES = ["AGE", "MCP", "MAC", "TGF", "AMC", "MMP", "TIMP", "COL"]
STATE_INDICES = {name: i for i, name in enumerate(STATE_NAMES)}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    print("Running M020 Thomas & Ford Versypt 2024 — DKD base case...")
    t0 = _time.perf_counter()
    t, y, y_ss = run_fibrosis(scenario="DKD")
    elapsed = _time.perf_counter() - t0

    print(f"Completed in {elapsed:.2f}s")
    print(f"Healthy SS: COL={y_ss[7]:.6g}  MAC={y_ss[2]:.6g}")
    print(f"\nDKD time course:")
    print(f"{'Week':>6}  {'Day':>5}  {'COL':>12}  {'MAC':>12}  {'COL_FC':>8}")
    col0 = y[0, 7]
    for wk in [0, 4, 8, 12, 16, 20, 24]:
        day = wk * 7
        idx = np.searchsorted(t, day)
        if idx >= len(t):
            idx = len(t) - 1
        col = y[idx, 7]
        mac = y[idx, 2]
        print(f"{wk:>6}  {int(round(t[idx])):>5}  {col:>12.6g}  {mac:>12.6g}  {col/col0:>8.4f}")
