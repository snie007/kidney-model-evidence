"""
Smith & Layton 2023 — Intrarenal RAS Model
Python port of the MATLAB model by N.H. Smith & A.T. Layton (2023).

Original MATLAB: model.m, get_pars.m, run_model.m
Solver: scipy.integrate.solve_ivp with LSODA (matches MATLAB ode15i stiff DAE solver)

PARAMETER VERSION NOTE (verified 2026-06-23):
  The MATLAB source code ported here is the 2021 bioRxiv preprint version
  (Smith & Layton, bioRxiv 2021.12.14.472639). The published journal article
  (J Math Biol 2023; DOI 10.1007/s00285-023-01891-y) underwent substantial
  re-fitting before publication. Consequently, steady-state AngII concentrations
  from this port differ from the paper's published Table 1:
    AngII_circ:   preprint=43.4 fmol/mL  vs paper=~14 fmol/mL  (3.1× off)
    AngII_Isf_Gl: preprint=40.9 fmol/mL  vs paper=~100 fmol/mL (2.4× off)
    AngII_Isf_Pt: preprint=189  fmol/mL  vs paper=~300 fmol/mL  (1.6× off)
    AngII_Fl_Tb:  preprint=757  fmol/mL  vs paper=~200 fmol/mL  (3.8× off)
  All Python ODEs and parameters exactly reproduce the MATLAB preprint code.
  This is NOT a port error. Contact authors for the final publication parameter set.

STRUCTURAL NOTE (2026-06-24):
  Two parameters, k_AngI_Gl and c_ACE_Gl, have been added to enable calibration
  to the 2023 published Table 1 targets.  In the 2021 preprint, glomerular ISF
  AngII is purely filtration-driven (AngII_Isf_Gl ≈ AngII_circ).  The 2023
  paper has AngII_Isf_Gl = 100 >> AngII_circ = 14, which requires local AngI
  synthesis + ACE in the glomerular ISF compartment.  With the default values
  k_AngI_Gl = 0 and c_ACE_Gl = 0, the model exactly reproduces the 2021
  preprint MATLAB output.  Run calibrate_to_paper2023.py to fit these parameters
  (plus v_max, k_AngI_Pt, k_AngI_Tb, V_Pt_Isf) to the 2023 Table 1 targets.

The MATLAB system is a DAE (differential-algebraic): some equations are purely
algebraic constraints (no x_p term).  We convert to explicit ODE by substituting
the algebraic relations directly.

State vector (40 variables, no-infusion baseline):
  x[0]  AGT_circ            fmol/mL
  x[1]  AngI_circ           fmol/mL
  x[2]  AngII_circ          fmol/mL
  x[3]  AT1R_AngII_memb_circ  fmol/mL
  x[4]  AT1R_memb_circ      fmol/mL   (algebraic: = AT1R_circ_tot - AT1R_AngII_memb_circ)
  x[5]  Ang17_circ          fmol/mL
  x[6]  PRC                 (dimensionless)
  x[7]  PRA                 fmol/mL/min (algebraic: = v_max*PRC*AGT/(AGT+K_M))
  x[8]  AngI_Isf_Gl         fmol/mL
  x[9]  AngII_Isf_Gl        fmol/mL
  x[10] AT1R_AngII_memb_Gl  fmol/mL
  x[11] AT1R_AngII_cell_Gl  fmol/mL
  x[12] AngII_cell_Gl       fmol/mL
  x[13] AT1R_memb_Gl        fmol/mL   (algebraic)
  x[14] AT1R_cell_Gl        fmol/mL
  x[15] AngI_Isf_Pt         fmol/mL
  x[16] AngII_Isf_Pt        fmol/mL
  x[17] AT1R_AngII_memb_Pt  fmol/mL
  x[18] AT1R_AngII_cell_Pt  fmol/mL
  x[19] AngII_cell_Pt       fmol/mL
  x[20] AT1R_memb_Pt        fmol/mL   (differential here — driven by fb_Pt)
  x[21] AT1R_cell_Pt        fmol/mL
  x[22] AngI_Fl_Tb          fmol/mL
  x[23] AngII_Fl_Tb         fmol/mL
  x[24] AT1R_AngII_memb_Tb  fmol/mL
  x[25] AT1R_AngII_cell_Tb  fmol/mL
  x[26] AngII_cell_Tb       fmol/mL
  x[27] AT1R_memb_Tb        fmol/mL   (differential — driven by fb_Tb)
  x[28] AT1R_cell_Tb        fmol/mL
  x[29] AngI_Pv             fmol/mL
  x[30] AngII_Pv            fmol/mL
  x[31] AT1R_AngII_memb_Pv  fmol/mL
  x[32] AT1R_memb_Pv        fmol/mL   (algebraic: = AT1R_Pv_tot - AT1R_AngII_memb_Pv)
  x[33] AngI_T              fmol       (algebraic)
  x[34] AngII_T             fmol       (algebraic)
  x[35] nu_AT1R             (algebraic feedback)
  x[36] fb_circ_AGT         (algebraic feedback)
  x[37] fb_circ_ACE         (algebraic feedback)
  x[38] fb_Pt               (algebraic feedback)
  x[39] fb_Tb               (algebraic feedback)

Note: x[4], x[7], x[13], x[32]-x[39] are algebraic but we track them in the
state vector for compatibility with the MATLAB output format.  Their ODEs are
set to zero (they are re-derived from algebra at every step).
"""

import numpy as np
from scipy.integrate import solve_ivp
from math import log

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def get_params():
    """Return parameter dict matching get_pars.m exactly."""
    p = {}

    # --- Fitted (baseline) parameters ---
    p['k_int']       = 0.193           # /min
    p['k_rec']       = (4/3) * 0.208   # /min
    p['k_lys']       = 0.208           # /min
    p['AT1R_Gl_tot'] = 115.0           # fmol/g kidney
    p['c_chym']      = 0.115           # /min
    p['c_ACE_circ']  = 9 * 0.115       # /min  (= 9*c_chym)
    p['c_NEP']       = 0.293           # /min
    p['c_ACE2']      = 0.21 * 0.293    # /min  (= 0.21*c_NEP)
    p['v_max']       = 99.7            # /min
    p['k_AngI_Gl']   = 0.0             # fmol/mL/min  (0 = 2021 preprint; fit for 2023 paper)
    p['c_ACE_Gl']    = 0.0             # /min          (0 = 2021 preprint; fit for 2023 paper)
    p['k_AngI_Pt']   = 6746.0          # fmol/mL/min
    p['c_ACE_Pt']    = 2.0             # /min
    p['k_AngI_Tb']   = 9992.0          # fmol/mL/min
    p['c_ACE_Tb']    = 1.83            # /min
    p['S_Tb']        = 0.241
    p['k_AGT']       = 1737.0          # fmol/mL/min

    # --- Literature parameters ---
    p['W_K']         = 1.49            # g kidney weight
    p['V_circ']      = 10.3            # mL

    # Renal volumes (mL per gram kidney — used per-kidney)
    p['V_Gl_Isf']    = 0.0019          # mL/g
    p['V_Gl_cell']   = 0.0019          # mL/g
    p['V_Pt_Isf']    = 0.0236          # mL/g
    p['V_Tb_Pt_cell']= 0.294           # mL/g
    p['V_Tb_Fl']     = 0.102           # mL/g
    p['V_Pv']        = 0.085           # mL/g

    # Renal hemodynamics
    p['phi_RPF_total'] = 11.6          # mL/min (total, will be divided by W_K in model)
    p['FF']           = 0.26

    # AT1R binding kinetics
    p['K_D']         = 1000.0          # fmol/mL
    p['k_ass']       = 2.4e-5          # mL/fmol/min
    p['k_diss']      = 2.4e-5 * 1000.0 # /min  (= k_ass * K_D)

    # Systemic half-lives -> degradation rates
    p['h_AGT']       = 240.0           # min
    p['h_renin']     = 3.0             # min
    p['h_AngI']      = 0.5             # min
    p['h_AngII']     = 0.267           # min
    p['h_Ang17']     = 0.167           # min
    p['v_AGT']       = log(2) / 240.0
    p['v_renin']     = log(2) / 3.0
    p['v_I']         = log(2) / 0.5
    p['v_II']        = log(2) / 0.267
    p['v_17']        = log(2) / 0.167
    p['R_sec']       = 1.0
    p['K_M']         = 2.8e6           # fmol/mL

    # Feedback parameters (note scaling as in get_pars.m + model.m unpacking)
    # model.m applies additional scaling during parameter-vector unpacking:
    #   line 65: K_circ_AGT = fb_pars(4) * 10^2   → ×100
    #   line 67: K_Tb       = fb_pars(6) * 10^(-1) → ×0.1
    scale = 1440.0 / 240.0
    p['k_a']          = 5.41 / scale / 240.0  # /min
    p['B_AT1R']       = 2.9
    p['K_circ_ACE']   = 3.9            # no extra scaling (model.m line 64, no multiplier)
    p['K_circ_AGT']   = 27.0 / scale * 1e2    # get_pars stores raw; model.m ×100 → 450.0
    p['K_Pt']         = 4.95 / scale           # no extra scaling (model.m line 66, no multiplier)
    p['K_Tb']         = 3.0  / scale * 1e-1   # get_pars stores raw; model.m ×0.1 → 0.05

    p['AT1R_circ_tot'] = 5.215e5       # fmol/mL
    p['AT1R_Pv_tot']   = 500.0         # fmol/mL

    # Steady-state AT1R-AngII complex values (for feedback normalisation)
    # Loaded from model_SS.mat in MATLAB; here we use the exact MATLAB values
    p['AT1R_AngII_memb_Gl_eq']   = 262.3520492
    p['AT1R_AngII_memb_Pt_eq']   = 88.62631451
    p['AT1R_AngII_memb_Tb_eq']   = 26.91758584
    p['AT1R_AngII_memb_circ_eq'] = 21679.03711

    # Derived flow rates (computed once from params)
    phi_RPF      = p['phi_RPF_total'] / p['W_K']  # per-gram RPF (mL/min/g)
    phi_GFR      = p['FF'] * phi_RPF
    phi_L        = (0.02 / 1.02) * phi_GFR
    phi_U        = phi_L
    phi_Pt       = p['S_Tb'] * (phi_GFR - phi_U)
    phi_Pv       = phi_GFR - phi_U

    p['phi_RPF'] = phi_RPF
    p['phi_GFR'] = phi_GFR
    p['phi_L']   = phi_L
    p['phi_U']   = phi_U
    p['phi_Pt']  = phi_Pt
    p['phi_Pv']  = phi_Pv

    return p


# ---------------------------------------------------------------------------
# Algebraic relationships (substitute into ODEs directly)
# ---------------------------------------------------------------------------

def _algebraic(x, p):
    """
    Compute algebraic quantities from the state vector.
    Returns a dict of derived quantities.
    """
    AT1R_AngII_memb_circ = x[3]
    AT1R_AngII_memb_Gl   = x[10]
    AT1R_AngII_cell_Gl   = x[11]
    AT1R_cell_Gl         = x[14]
    AT1R_AngII_memb_Pt   = x[17]
    AT1R_AngII_memb_Tb   = x[24]
    AT1R_AngII_memb_Pv   = x[31]

    AGT_circ = x[0]
    PRC      = x[6]

    a = {}

    # Algebraic: free membrane receptors
    a['AT1R_memb_circ'] = p['AT1R_circ_tot'] - AT1R_AngII_memb_circ
    a['AT1R_memb_Gl']   = (p['AT1R_Gl_tot'] / p['V_Gl_Isf']
                           - AT1R_AngII_memb_Gl
                           - (p['V_Gl_cell'] / p['V_Gl_Isf']) * (AT1R_AngII_cell_Gl + AT1R_cell_Gl))
    a['AT1R_memb_Pv']   = p['AT1R_Pv_tot'] - AT1R_AngII_memb_Pv

    # PRA (Michaelis-Menten)
    a['PRA'] = p['v_max'] * PRC * AGT_circ / (AGT_circ + p['K_M'])

    # Feedback ratios
    q_circ = AT1R_AngII_memb_circ / p['AT1R_AngII_memb_circ_eq']
    q_Gl   = AT1R_AngII_memb_Gl   / p['AT1R_AngII_memb_Gl_eq']
    q_Pt   = AT1R_AngII_memb_Pt   / p['AT1R_AngII_memb_Pt_eq']
    q_Tb   = AT1R_AngII_memb_Tb   / p['AT1R_AngII_memb_Tb_eq']

    a['q_circ'] = q_circ
    a['q_Gl']   = q_Gl
    a['q_Pt']   = q_Pt
    a['q_Tb']   = q_Tb

    # nu_AT1R (renin secretion feedback)
    a['nu_AT1R'] = q_Gl ** (-p['B_AT1R'])

    # Liver AGT and systemic ACE feedback (only active if q_circ > 1)
    a['fb_circ_AGT'] = p['K_circ_AGT'] * (q_circ - 1) if q_circ > 1 else 0.0
    a['fb_circ_ACE'] = p['K_circ_ACE'] * (q_circ - 1) if q_circ > 1 else 0.0

    # Basolateral AT1R expression feedback (q_Pt > 1)
    a['fb_Pt'] = p['K_Pt'] * (q_Pt - 1) if q_Pt > 1 else 0.0

    # Apical AT1R expression feedback
    # Note: MATLAB bug-compatible — uses q_Pt > 1 for fb_Tb check too
    a['fb_Tb'] = p['K_Tb'] * (q_Tb - 1) if q_Pt > 1 else 0.0

    return a


# ---------------------------------------------------------------------------
# Initial conditions (from model_SS.mat — MATLAB reference values)
# ---------------------------------------------------------------------------

MATLAB_SS = np.array([
    575965.2099,    # x[0]  AGT_circ
    65.27775088,    # x[1]  AngI_circ
    43.37360519,    # x[2]  AngII_circ
    21679.03711,    # x[3]  AT1R_AngII_memb_circ
    499820.9629,    # x[4]  AT1R_memb_circ  (algebraic)
    5.251109208,    # x[5]  Ang17_circ
    4.328085123,    # x[6]  PRC
    73.61888593,    # x[7]  PRA              (algebraic)
    65.27775088,    # x[8]  AngI_Isf_Gl
    40.94967293,    # x[9]  AngII_Isf_Gl
    262.3520492,    # x[10] AT1R_AngII_memb_Gl
    2154.192239,    # x[11] AT1R_AngII_cell_Gl
    243.4324302,    # x[12] AngII_cell_Gl
    57927.19718,    # x[13] AT1R_memb_Gl     (algebraic)
    182.5743226,    # x[14] AT1R_cell_Gl
    454.1318055,    # x[15] AngI_Isf_Pt
    189.3315007,    # x[16] AngII_Isf_Pt
    88.62631451,    # x[17] AT1R_AngII_memb_Pt
    57.24287684,    # x[18] AT1R_AngII_cell_Pt
    6.601176369,    # x[19] AngII_cell_Pt
    4232.415582,    # x[20] AT1R_memb_Pt
    4.950882277,    # x[21] AT1R_cell_Pt
    1633.98394,     # x[22] AngI_Fl_Tb
    757.3451171,    # x[23] AngII_Fl_Tb
    26.91758584,    # x[24] AT1R_AngII_memb_Tb
    75.15546431,    # x[25] AT1R_AngII_cell_Tb
    8.665286414,    # x[26] AngII_cell_Tb
    321.3592233,    # x[27] AT1R_memb_Tb
    6.49896481,     # x[28] AT1R_cell_Tb
    163.7577761,    # x[29] AngI_Pv
    79.10401876,    # x[30] AngII_Pv
    36.65263838,    # x[31] AT1R_AngII_memb_Pv
    463.3473616,    # x[32] AT1R_memb_Pv     (algebraic)
    191.4273111,    # x[33] AngI_T            (algebraic)
    144.9391285,    # x[34] AngII_T           (algebraic)
    1.0,            # x[35] nu_AT1R           (algebraic)
    0.0,            # x[36] fb_circ_AGT       (algebraic)
    0.0,            # x[37] fb_circ_ACE       (algebraic)
    0.0,            # x[38] fb_Pt             (algebraic)
    0.0,            # x[39] fb_Tb             (algebraic)
])

# MATLAB reference: 7-day baseline final values (from run_baseline_check.m)
MATLAB_REF_7DAY = {
    'AGT_circ':           575940.9215,
    'AngI_circ':          65.27724329,
    'AngII_circ':         43.37337613,
    'AT1R_AngII_memb_circ': 21678.92738,
    'AT1R_memb_circ':     499821.0726,
    'Ang17_circ':         5.25106998,
    'PRC':                4.328148671,
    'PRA':                73.61739195,
    'AngI_Isf_Gl':        65.27724329,
    'AngII_Isf_Gl':       40.94945614,
    'AT1R_AngII_memb_Gl': 262.3507209,
    'AT1R_AngII_cell_Gl': 2154.181109,
    'AngII_cell_Gl':      243.4311978,
    'AT1R_memb_Gl':       57927.21056,
    'AT1R_cell_Gl':       182.5733983,
    'AngI_Isf_Pt':        454.1314689,
    'AngII_Isf_Pt':       189.3311577,
    'AT1R_AngII_memb_Pt': 88.62618118,
    'AT1R_AngII_cell_Pt': 57.24279068,
    'AngII_cell_Pt':      6.601166438,
    'AT1R_memb_Pt':       4232.416882,
    'AT1R_cell_Pt':       4.950874829,
    'AngI_Fl_Tb':         1633.982481,
    'AngII_Fl_Tb':        757.3436975,
    'AT1R_AngII_memb_Tb': 26.91755806,
    'AT1R_AngII_cell_Tb': 75.15538671,
    'AngII_cell_Tb':      8.665277473,
    'AT1R_memb_Tb':       321.3594941,
    'AT1R_cell_Tb':       6.498958105,
    'AngI_Pv':            163.7573171,
    'AngII_Pv':           79.10376629,
    'AT1R_AngII_memb_Pv': 36.65252998,
    'AT1R_memb_Pv':       463.34747,
    'AngI_T':             191.4271145,
    'AngII_T':            144.9388588,
    'nu_AT1R':            1.000014683,
    'fb_circ_AGT':        0.0,
    'fb_circ_ACE':        0.0,
    'fb_Pt':              0.0,
    'fb_Tb':              0.0,
}

STATE_NAMES = [
    'AGT_circ', 'AngI_circ', 'AngII_circ', 'AT1R_AngII_memb_circ',
    'AT1R_memb_circ', 'Ang17_circ', 'PRC', 'PRA',
    'AngI_Isf_Gl', 'AngII_Isf_Gl', 'AT1R_AngII_memb_Gl',
    'AT1R_AngII_cell_Gl', 'AngII_cell_Gl', 'AT1R_memb_Gl', 'AT1R_cell_Gl',
    'AngI_Isf_Pt', 'AngII_Isf_Pt', 'AT1R_AngII_memb_Pt',
    'AT1R_AngII_cell_Pt', 'AngII_cell_Pt', 'AT1R_memb_Pt', 'AT1R_cell_Pt',
    'AngI_Fl_Tb', 'AngII_Fl_Tb', 'AT1R_AngII_memb_Tb',
    'AT1R_AngII_cell_Tb', 'AngII_cell_Tb', 'AT1R_memb_Tb', 'AT1R_cell_Tb',
    'AngI_Pv', 'AngII_Pv', 'AT1R_AngII_memb_Pv', 'AT1R_memb_Pv',
    'AngI_T', 'AngII_T',
    'nu_AT1R', 'fb_circ_AGT', 'fb_circ_ACE', 'fb_Pt', 'fb_Tb',
]


def initial_conditions(p=None):
    """Return the steady-state initial conditions (from model_SS.mat)."""
    return MATLAB_SS.copy()


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def odes(t, x, p, K_AngII=0.0):
    """
    Explicit ODE for the Smith & Layton 2023 intrarenal RAS model.

    The MATLAB model is a DAE (ode15i).  Here the algebraic constraints are
    substituted analytically, giving an explicit ODE for the 40-component
    state vector.

    Algebraic variables (x[4], x[7], x[13], x[32]-x[39]) have dxdt = 0 here
    but are kept in the state vector for output compatibility; they are
    re-derived from the differential states at each call to avoid drift.

    Parameters
    ----------
    t : float
        Time (minutes)
    x : array_like (40,)
        State vector
    p : dict
        Parameters from get_params()
    K_AngII : float
        Exogenous Ang II infusion rate (fmol/mL/min), default 0.

    Returns
    -------
    dxdt : np.ndarray (40,)
    """
    dxdt = np.zeros(40)

    # Unpack state
    AGT_circ             = x[0]
    AngI_circ            = x[1]
    AngII_circ           = x[2]
    AT1R_AngII_memb_circ = x[3]
    # x[4] AT1R_memb_circ  — algebraic, derived below
    Ang17_circ           = x[5]
    PRC                  = x[6]
    # x[7] PRA             — algebraic, derived below
    AngI_Isf_Gl          = x[8]
    AngII_Isf_Gl         = x[9]
    AT1R_AngII_memb_Gl   = x[10]
    AT1R_AngII_cell_Gl   = x[11]
    AngII_cell_Gl        = x[12]
    # x[13] AT1R_memb_Gl   — algebraic
    AT1R_cell_Gl         = x[14]
    AngI_Isf_Pt          = x[15]
    AngII_Isf_Pt         = x[16]
    AT1R_AngII_memb_Pt   = x[17]
    AT1R_AngII_cell_Pt   = x[18]
    AngII_cell_Pt        = x[19]
    AT1R_memb_Pt         = x[20]
    AT1R_cell_Pt         = x[21]
    AngI_Fl_Tb           = x[22]
    AngII_Fl_Tb          = x[23]
    AT1R_AngII_memb_Tb   = x[24]
    AT1R_AngII_cell_Tb   = x[25]
    AngII_cell_Tb        = x[26]
    AT1R_memb_Tb         = x[27]
    AT1R_cell_Tb         = x[28]
    AngI_Pv              = x[29]
    AngII_Pv             = x[30]
    AT1R_AngII_memb_Pv   = x[31]
    # x[32] AT1R_memb_Pv   — algebraic
    # x[33] AngI_T         — algebraic
    # x[34] AngII_T        — algebraic
    # x[35]-x[39]          — algebraic feedbacks

    # Unpack params
    k_int        = p['k_int']
    k_rec        = p['k_rec']
    k_lys        = p['k_lys']
    AT1R_Gl_tot  = p['AT1R_Gl_tot']
    c_chym       = p['c_chym']
    c_ACE_circ   = p['c_ACE_circ']
    c_NEP        = p['c_NEP']
    c_ACE2       = p['c_ACE2']
    v_max        = p['v_max']
    k_AngI_Pt    = p['k_AngI_Pt']
    c_ACE_Pt     = p['c_ACE_Pt']
    k_AngI_Tb    = p['k_AngI_Tb']
    c_ACE_Tb     = p['c_ACE_Tb']
    k_AGT        = p['k_AGT']
    W_K          = p['W_K']
    V_circ       = p['V_circ']
    V_Gl_Isf     = p['V_Gl_Isf']
    V_Gl_cell    = p['V_Gl_cell']
    V_Pt_Isf     = p['V_Pt_Isf']
    V_Tb_Pt_cell = p['V_Tb_Pt_cell']
    V_Tb_Fl      = p['V_Tb_Fl']
    V_Pv         = p['V_Pv']
    phi_RPF      = p['phi_RPF']
    phi_GFR      = p['phi_GFR']
    phi_L        = p['phi_L']
    phi_U        = p['phi_U']
    phi_Pt       = p['phi_Pt']
    phi_Pv       = p['phi_Pv']
    k_ass        = p['k_ass']
    k_diss       = p['k_diss']
    v_AGT        = p['v_AGT']
    v_renin      = p['v_renin']
    v_I          = p['v_I']
    v_II         = p['v_II']
    v_17         = p['v_17']
    R_sec        = p['R_sec']
    K_M          = p['K_M']
    K_circ_ACE   = p['K_circ_ACE']
    K_circ_AGT   = p['K_circ_AGT']
    K_Pt         = p['K_Pt']
    K_Tb         = p['K_Tb']
    B_AT1R       = p['B_AT1R']
    AT1R_circ_tot = p['AT1R_circ_tot']
    AT1R_Pv_tot  = p['AT1R_Pv_tot']

    # --- Algebraic quantities ---
    AT1R_memb_circ = AT1R_circ_tot - AT1R_AngII_memb_circ
    AT1R_memb_Gl   = (AT1R_Gl_tot / V_Gl_Isf
                      - AT1R_AngII_memb_Gl
                      - (V_Gl_cell / V_Gl_Isf) * (AT1R_AngII_cell_Gl + AT1R_cell_Gl))
    AT1R_memb_Pv   = AT1R_Pv_tot - AT1R_AngII_memb_Pv
    PRA            = v_max * PRC * AGT_circ / (AGT_circ + K_M)

    # Feedback ratios
    q_circ = AT1R_AngII_memb_circ / p['AT1R_AngII_memb_circ_eq']
    q_Gl   = AT1R_AngII_memb_Gl   / p['AT1R_AngII_memb_Gl_eq']
    q_Pt   = AT1R_AngII_memb_Pt   / p['AT1R_AngII_memb_Pt_eq']
    q_Tb   = AT1R_AngII_memb_Tb   / p['AT1R_AngII_memb_Tb_eq']

    # nu_AT1R (renin secretion modulation)
    nu_AT1R = q_Gl ** (-B_AT1R)

    # Systemic feedbacks (only if q > 1)
    fb_circ_AGT = K_circ_AGT * (q_circ - 1) if q_circ > 1 else 0.0
    fb_circ_ACE = K_circ_ACE * (q_circ - 1) if q_circ > 1 else 0.0

    # Intrarenal feedbacks
    fb_Pt = K_Pt * (q_Pt - 1) if q_Pt > 1 else 0.0
    # MATLAB: fb_Tb uses q_Pt > 1 as gate (reproducing MATLAB as-is)
    fb_Tb = K_Tb * (q_Tb - 1) if q_Pt > 1 else 0.0

    # Total effective ACE in circulation (basal + feedback)
    c_ACE_eff = c_ACE_circ + fb_circ_ACE

    # ========== SYSTEMIC ==========

    # x[0]: AGT_circ
    dxdt[0] = k_AGT + fb_circ_AGT - PRA - v_AGT * AGT_circ

    # x[1]: AngI_circ
    dxdt[1] = (PRA
               - (c_chym + c_ACE_eff + c_NEP + v_I) * AngI_circ
               + (W_K / V_circ) * phi_L * (AngI_Isf_Pt + AngI_Isf_Gl)
               + (W_K / V_circ) * ((phi_RPF - 2*phi_L - phi_U) * AngI_Pv
                                   - phi_RPF * AngI_circ))

    # x[2]: AngII_circ
    dxdt[2] = (K_AngII
               + (c_chym + c_ACE_eff) * AngI_circ
               - (c_ACE2 + v_II) * AngII_circ
               + (W_K / V_circ) * phi_L * (AngII_Isf_Pt + AngII_Isf_Gl)
               + (W_K / V_circ) * (phi_RPF - 2*phi_L - phi_U) * AngII_Pv
               - (W_K / V_circ) * phi_RPF * AngII_circ
               + k_diss * AT1R_AngII_memb_circ
               - k_ass * AT1R_memb_circ * AngII_circ)

    # x[3]: AT1R_AngII_memb_circ
    dxdt[3] = (k_ass * AT1R_memb_circ * AngII_circ
               - k_diss * AT1R_AngII_memb_circ)

    # x[4]: AT1R_memb_circ — algebraic; keep at 0 derivative, update via algebraic
    dxdt[4] = 0.0

    # x[5]: Ang17_circ
    dxdt[5] = c_NEP * AngI_circ + c_ACE2 * AngII_circ - v_17 * Ang17_circ

    # x[6]: PRC
    dxdt[6] = R_sec * nu_AT1R - v_renin * PRC

    # x[7]: PRA — algebraic
    dxdt[7] = 0.0

    # ========== GLOMERULAR ==========
    k_AngI_Gl = p['k_AngI_Gl']   # local AngI synthesis (0 in 2021 preprint)
    c_ACE_Gl  = p['c_ACE_Gl']    # local ACE activity   (0 in 2021 preprint)

    # x[8]: AngI_Isf_Gl
    dxdt[8] = ((phi_L / V_Gl_Isf) * (AngI_circ - AngI_Isf_Gl)
               + k_AngI_Gl
               - c_ACE_Gl * AngI_Isf_Gl)

    # x[9]: AngII_Isf_Gl
    dxdt[9] = ((phi_L / V_Gl_Isf) * (AngII_circ - AngII_Isf_Gl)
               + c_ACE_Gl * AngI_Isf_Gl
               + k_diss * AT1R_AngII_memb_Gl
               - k_ass * AngII_Isf_Gl * AT1R_memb_Gl)

    # x[10]: AT1R_AngII_memb_Gl
    dxdt[10] = (k_ass * AngII_Isf_Gl * AT1R_memb_Gl
                - (k_diss + k_int) * AT1R_AngII_memb_Gl)

    # x[11]: AT1R_AngII_cell_Gl
    dxdt[11] = ((V_Gl_Isf / V_Gl_cell) * k_int * AT1R_AngII_memb_Gl
                + k_ass * AngII_cell_Gl * AT1R_cell_Gl
                - k_diss * AT1R_AngII_cell_Gl)

    # x[12]: AngII_cell_Gl
    dxdt[12] = (k_diss * AT1R_AngII_cell_Gl
                - k_ass * AngII_cell_Gl * AT1R_cell_Gl
                - k_lys * AngII_cell_Gl)

    # x[13]: AT1R_memb_Gl — algebraic
    dxdt[13] = 0.0

    # x[14]: AT1R_cell_Gl
    dxdt[14] = (k_diss * AT1R_AngII_cell_Gl
                - k_ass * AngII_cell_Gl * AT1R_cell_Gl
                - k_rec * AT1R_cell_Gl)

    # ========== PERITUBULAR ==========

    # x[15]: AngI_Isf_Pt
    dxdt[15] = (k_AngI_Pt
                + (phi_Pt / V_Pt_Isf) * AngI_Fl_Tb
                - (c_ACE_Pt + (phi_Pv + phi_L) / V_Pt_Isf) * AngI_Isf_Pt)

    # x[16]: AngII_Isf_Pt
    dxdt[16] = (c_ACE_Pt * AngI_Isf_Pt
                + (phi_Pt / V_Pt_Isf) * AngII_Fl_Tb
                - ((phi_Pv + phi_L) / V_Pt_Isf) * AngII_Isf_Pt
                + k_diss * AT1R_AngII_memb_Pt
                - k_ass * AngII_Isf_Pt * AT1R_memb_Pt)

    # x[17]: AT1R_AngII_memb_Pt
    dxdt[17] = (k_ass * AngII_Isf_Pt * AT1R_memb_Pt
                - (k_diss + k_int) * AT1R_AngII_memb_Pt)

    # x[18]: AT1R_AngII_cell_Pt
    dxdt[18] = ((V_Pt_Isf / V_Tb_Pt_cell) * k_int * AT1R_AngII_memb_Pt
                + k_ass * AngII_cell_Pt * AT1R_cell_Pt
                - k_diss * AT1R_AngII_cell_Pt)

    # x[19]: AngII_cell_Pt
    dxdt[19] = (k_diss * AT1R_AngII_cell_Pt
                - k_ass * AngII_cell_Pt * AT1R_cell_Pt
                - k_lys * AngII_cell_Pt)

    # x[20]: AT1R_memb_Pt  (differential in MATLAB)
    dxdt[20] = (fb_Pt
                + (V_Tb_Pt_cell / V_Pt_Isf) * k_rec * AT1R_cell_Pt
                + k_diss * AT1R_AngII_memb_Pt
                - k_ass * AngII_Isf_Pt * AT1R_memb_Pt)

    # x[21]: AT1R_cell_Pt
    dxdt[21] = (k_diss * AT1R_AngII_cell_Pt
                - k_ass * AngII_cell_Pt * AT1R_cell_Pt
                - k_rec * AT1R_cell_Pt)

    # ========== TUBULAR ==========

    # x[22]: AngI_Fl_Tb
    dxdt[22] = (k_AngI_Tb
                + (phi_GFR / V_Tb_Fl) * AngI_circ
                - (c_ACE_Tb + (phi_U + phi_Pt) / V_Tb_Fl) * AngI_Fl_Tb)

    # x[23]: AngII_Fl_Tb
    dxdt[23] = ((phi_GFR / V_Tb_Fl) * AngII_circ
                + c_ACE_Tb * AngI_Fl_Tb
                + k_diss * AT1R_AngII_memb_Tb
                - k_ass * AngII_Fl_Tb * AT1R_memb_Tb
                - ((phi_U + phi_Pt) / V_Tb_Fl) * AngII_Fl_Tb)

    # x[24]: AT1R_AngII_memb_Tb
    dxdt[24] = (k_ass * AngII_Fl_Tb * AT1R_memb_Tb
                - (k_diss + k_int) * AT1R_AngII_memb_Tb)

    # x[25]: AT1R_AngII_cell_Tb
    dxdt[25] = ((V_Tb_Fl / V_Tb_Pt_cell) * k_int * AT1R_AngII_memb_Tb
                + k_ass * AngII_cell_Tb * AT1R_cell_Tb
                - k_diss * AT1R_AngII_cell_Tb)

    # x[26]: AngII_cell_Tb
    dxdt[26] = (k_diss * AT1R_AngII_cell_Tb
                - k_ass * AngII_cell_Tb * AT1R_cell_Tb
                - k_lys * AngII_cell_Tb)

    # x[27]: AT1R_memb_Tb (differential in MATLAB)
    dxdt[27] = (fb_Tb
                + (V_Tb_Pt_cell / V_Tb_Fl) * k_rec * AT1R_cell_Tb
                + k_diss * AT1R_AngII_memb_Tb
                - k_ass * AngII_Fl_Tb * AT1R_memb_Tb)

    # x[28]: AT1R_cell_Tb
    dxdt[28] = (k_diss * AT1R_AngII_cell_Tb
                - k_ass * AngII_cell_Tb * AT1R_cell_Tb
                - k_rec * AT1R_cell_Tb)

    # ========== RENAL VASCULATURE ==========

    # x[29]: AngI_Pv
    dxdt[29] = (((phi_RPF - phi_GFR - phi_L) / V_Pv) * AngI_circ
                + (phi_Pv / V_Pv) * AngI_Isf_Pt
                - (((phi_RPF - 2*phi_L - phi_U) / V_Pv) + v_I) * AngI_Pv)

    # x[30]: AngII_Pv
    dxdt[30] = (((phi_RPF - phi_GFR - phi_L) / V_Pv) * AngII_circ
                + (phi_Pv / V_Pv) * AngII_Isf_Pt
                - (((phi_RPF - 2*phi_L - phi_U) / V_Pv) + v_II) * AngII_Pv
                - k_ass * AT1R_memb_Pv * AngII_Pv
                + k_diss * AT1R_AngII_memb_Pv)

    # x[31]: AT1R_AngII_memb_Pv
    dxdt[31] = (k_ass * AT1R_memb_Pv * AngII_Pv
                - k_diss * AT1R_AngII_memb_Pv)

    # x[32]: AT1R_memb_Pv — algebraic
    dxdt[32] = 0.0

    # ========== ALGEBRAIC / WHOLE KIDNEY ==========

    # x[33]: AngI_T — algebraic summation
    dxdt[33] = 0.0

    # x[34]: AngII_T — algebraic summation
    dxdt[34] = 0.0

    # x[35]-x[39]: feedback variables — algebraic
    dxdt[35] = 0.0  # nu_AT1R
    dxdt[36] = 0.0  # fb_circ_AGT
    dxdt[37] = 0.0  # fb_circ_ACE
    dxdt[38] = 0.0  # fb_Pt
    dxdt[39] = 0.0  # fb_Tb

    return dxdt


def _update_algebraic(x, p):
    """
    After ODE integration, update the algebraic variables in-place.
    Call on each time slice of the solution array.
    """
    AT1R_AngII_memb_circ = x[3]
    AT1R_AngII_memb_Gl   = x[10]
    AT1R_AngII_cell_Gl   = x[11]
    AT1R_cell_Gl         = x[14]
    AT1R_AngII_memb_Pv   = x[31]
    AGT_circ             = x[0]
    PRC                  = x[6]
    AT1R_AngII_memb_Pt   = x[17]
    AT1R_AngII_memb_Tb   = x[24]

    AngI_Isf_Gl  = x[8]
    AngI_Isf_Pt  = x[15]
    AngI_Fl_Tb   = x[22]
    AngI_Pv      = x[29]
    AngII_Isf_Gl = x[9]
    AngII_Isf_Pt = x[16]
    AngII_Fl_Tb  = x[23]
    AT1R_AngII_memb_Gl_  = x[10]
    AT1R_AngII_memb_Pt_  = x[17]
    AT1R_AngII_memb_Tb_  = x[24]
    AT1R_AngII_cell_Gl_  = x[11]
    AT1R_AngII_cell_Pt   = x[18]
    AT1R_AngII_cell_Tb   = x[25]
    AngII_cell_Gl        = x[12]
    AngII_cell_Pt        = x[19]
    AngII_cell_Tb        = x[26]
    AngII_Pv             = x[30]
    AT1R_AngII_memb_Pv_  = x[31]

    V_Gl_Isf     = p['V_Gl_Isf']
    V_Gl_cell    = p['V_Gl_cell']
    V_Pt_Isf     = p['V_Pt_Isf']
    V_Tb_Pt_cell = p['V_Tb_Pt_cell']
    V_Tb_Fl      = p['V_Tb_Fl']
    V_Pv         = p['V_Pv']

    # x[4]: AT1R_memb_circ
    x[4]  = p['AT1R_circ_tot'] - AT1R_AngII_memb_circ

    # x[7]: PRA
    x[7]  = p['v_max'] * PRC * AGT_circ / (AGT_circ + p['K_M'])

    # x[13]: AT1R_memb_Gl
    x[13] = (p['AT1R_Gl_tot'] / V_Gl_Isf
             - AT1R_AngII_memb_Gl
             - (V_Gl_cell / V_Gl_Isf) * (AT1R_AngII_cell_Gl + AT1R_cell_Gl))

    # x[32]: AT1R_memb_Pv
    x[32] = p['AT1R_Pv_tot'] - AT1R_AngII_memb_Pv

    # x[33]: AngI_T
    x[33] = (V_Gl_Isf * AngI_Isf_Gl + V_Pt_Isf * AngI_Isf_Pt
             + V_Tb_Fl * AngI_Fl_Tb + V_Pv * AngI_Pv)

    # x[34]: AngII_T
    x[34] = (V_Gl_Isf * AngII_Isf_Gl
             + V_Pt_Isf * AngII_Isf_Pt
             + V_Tb_Fl * AngII_Fl_Tb
             + V_Gl_Isf * AT1R_AngII_memb_Gl
             + V_Pt_Isf * AT1R_AngII_memb_Pt
             + V_Tb_Fl * AT1R_AngII_memb_Tb
             + V_Tb_Pt_cell * (AT1R_AngII_cell_Tb + AT1R_AngII_cell_Pt
                               + AngII_cell_Tb + AngII_cell_Pt)
             + V_Gl_cell * (AT1R_AngII_cell_Gl + AngII_cell_Gl)
             + V_Pv * (AngII_Pv + AT1R_AngII_memb_Pv))

    # Feedback ratios
    q_circ = AT1R_AngII_memb_circ / p['AT1R_AngII_memb_circ_eq']
    q_Gl   = AT1R_AngII_memb_Gl   / p['AT1R_AngII_memb_Gl_eq']
    q_Pt   = AT1R_AngII_memb_Pt   / p['AT1R_AngII_memb_Pt_eq']
    q_Tb   = AT1R_AngII_memb_Tb   / p['AT1R_AngII_memb_Tb_eq']

    # x[35]: nu_AT1R
    x[35] = q_Gl ** (-p['B_AT1R'])

    # x[36]: fb_circ_AGT
    x[36] = p['K_circ_AGT'] * (q_circ - 1) if q_circ > 1 else 0.0

    # x[37]: fb_circ_ACE
    x[37] = p['K_circ_ACE'] * (q_circ - 1) if q_circ > 1 else 0.0

    # x[38]: fb_Pt
    x[38] = p['K_Pt'] * (q_Pt - 1) if q_Pt > 1 else 0.0

    # x[39]: fb_Tb
    x[39] = p['K_Tb'] * (q_Tb - 1) if q_Pt > 1 else 0.0

    return x


# ---------------------------------------------------------------------------
# Run baseline
# ---------------------------------------------------------------------------

def run_baseline(p=None, days=7, rtol=1e-8, atol=1e-10):
    """
    Integrate to steady state (7 days, no Ang II infusion) and return
    a dict of key output values.

    Parameters
    ----------
    p : dict or None
        Parameters from get_params(). If None, uses defaults.
    days : float
        Duration in days (default 7).
    rtol, atol : float
        Solver tolerances.

    Returns
    -------
    result : dict
        Keys are STATE_NAMES, values are final-time concentrations.
    sol : OdeResult
        Full scipy solution object.
    """
    if p is None:
        p = get_params()

    x0 = initial_conditions(p)
    t_end = days * 1440.0  # convert days to minutes

    sol = solve_ivp(
        fun=lambda t, x: odes(t, x, p, K_AngII=0.0),
        t_span=(0.0, t_end),
        y0=x0,
        method='LSODA',
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Extract final state and update algebraic variables
    x_final = sol.y[:, -1].copy()
    x_final = _update_algebraic(x_final, p)

    result = {name: x_final[i] for i, name in enumerate(STATE_NAMES)}
    return result, sol


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    print("Smith & Layton 2023 — Intrarenal RAS Model")
    print("Running 7-day baseline...")
    p = get_params()
    result, sol = run_baseline(p, days=7)
    print(f"\nSolver: {sol.message}")
    print(f"Steps:  {len(sol.t)}")
    print("\nKey steady-state values:")
    keys = ['AGT_circ', 'AngI_circ', 'AngII_circ', 'Ang17_circ', 'PRC', 'PRA',
            'AngII_Isf_Gl', 'AngII_Isf_Pt', 'AngII_Fl_Tb', 'AngII_Pv',
            'AngI_T', 'AngII_T']
    for k in keys:
        print(f"  {k:30s} = {result[k]:.6f}")
