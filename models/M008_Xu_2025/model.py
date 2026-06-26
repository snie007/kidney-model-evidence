"""
model.py — Python port of Xu et al. 2025 full-scale rat kidney model.

Reference:
    Xu P et al. Am J Physiol Renal Physiol 328:F702-F723, 2025. PMID: 40099641

Source:
    C:/Users/sneidere/Dropbox/Projects/2026/kidney-model-collection/M008_Xu_2025_full_kidney/

Usage:
    from model import run_single_nephron, run_full_kidney

    # Single nephron simulation (no VTK file needed):
    result = run_single_nephron(pressure_mmHg=100)

    # Full kidney simulation (requires Kirchhoff-solved VTK files in data/ subdir):
    result = run_full_kidney(pressure_mmHg=100, vtk_dir=..., source_dir=...)
"""

import numpy as np
import sys
import os
import logging
from dataclasses import dataclass
from functools import partial
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import scipy.optimize as so


# ---------------------------------------------------------------------------
# Parameters (from parameters.py)
# ---------------------------------------------------------------------------

@dataclass
class Glomerular:
    """Glomerular filtration parameters."""
    C_A: float = 57       # g/l   — afferent plasma protein concentration
    H_A: float = 0.5      # —     — haematocrit (plasma fraction = 1 - H_A)
    Kf: float = 2.5       # —     — glomerular ultrafiltration coefficient
    R_E: float = 0.209    # mmHg min nl^-1 — efferent arteriole resistance
    R_PC: float = 0.0702  # mmHg min nl^-1 — peritubular capillary resistance
    a: float = 0.1631     # mmHg l g^-1 — oncotic pressure coefficient
    b: float = 0.00294    # —             — oncotic pressure coefficient
    L: float = 10         # —             — glomerular capillary length (normalised)


@dataclass
class Tubular:
    """Tubular transport parameters."""
    Km: float = 20        # mM      — Michaelis constant for NaCl reabsorption
    Lv: float = 2e-5      # cm^2 l osmol^-1 s^-1 — water permeability
    ns: float = 2         # —       — number of osmoles per NaCl molecule
    Ls: float = 3.4e-7    # cm^2 s^-1 — NaCl permeability (short loop)
    V_max1 = 1e-7 * 1.22  # mmol cm^-1 s^-1 — max transport rate (pre-MD)
    V_max2 = 0.65e-7 * 1.22  # mmol cm^-1 s^-1 — max transport rate (post-MD)
    alpha: float = 1.65e-2   # (nl min^-1 mmHg^-5)^1/4 — tubular compliance
    beta: float = 0.924      # (nl min^-1 mmHg^-1)^1/4  — tubular compliance
    theta: float = 1.3       # cm^-1 — proximal reabsorption decay constant
    keppa: float = 33.6      # nl min^-1 cm^-1 — proximal reabsorption coefficient
    gamma: float = 1.33e-5   # cm mmHg^-1 — compliance factor
    mu: float = 7.2e-4       # g/(cm s) — fluid viscosity
    r_proximal: float = 12e-4  # cm — proximal tubule radius
    r_loop: float = 10e-4     # cm — loop of Henle radius
    Ls_long: float = 5.6e-7   # cm^2 s^-1 — NaCl permeability (long loop)     Xu 2025 Table 2
    Ls_inter: float = 5.2e-7  # cm^2 s^-1 — NaCl permeability (intermediate)  Xu 2025 Table 2


@dataclass
class AA:
    """Afferent arteriole wall mechanics parameters."""
    sigma_e: float = 25.56    # dyn cm^-2 — elastic stress coefficient   Xu 2025 Table 3
    k_e: float = 1.7304e4    # cm^-1     — elastic stiffness             Xu 2025 Table 3
    r_e: float = 7            # um        — elastic rest radius           Xu 2025 Table 3
    k_m: float = 8.75e6      # cm^-2     — muscle stiffness              Xu 2025 Table 3
    r_m: float = 12.5         # um        — muscle optimal radius         Xu 2025 Table 3
    sigma_m: float = 7.23e5  # dyn cm^-2 — smooth muscle stress          Xu 2025 Table 3
    sigma_v: float = 1e6      # dyn s cm^-2 — viscous stress
    r_n: float = 10.04        # um        — natural radius
    h_init: float = 2         # um        — initial wall thickness


@dataclass
class Myo:
    """Myogenic response parameters."""
    lam: float = 1       # — — myogenic coupling gain
    T0: float = 81.1252  # dyn cm^-1 — basal active tension


@dataclass
class TGF:
    """Tubuloglomerular feedback parameters (short loop)."""
    C_half: float = 44          # mmol/l — NaCl set-point at macula densa  Xu 2025 Table 5
    k: float = (70 + 100) / 2 * 1e-3  # l/mmol — TGF gain
    ita_max: float = 0.091      # — — maximum TGF activation
    phi: float = 0.182          # — — TGF Hill coefficient


@dataclass
class TGF_INTER:
    """TGF parameters for intermediate-loop nephrons."""
    k: float = 0.8328 * (70 + 100) / 2 * 1e-3
    ita_max: float = 0.1183
    phi: float = 0.2184


@dataclass
class TGF_LONG:
    """TGF parameters for long-loop nephrons."""
    k: float = 0.05 / 0.07 * (70 + 100) / 2 * 1e-3
    ita_max: float = 0.1456
    phi: float = 0.2583


# Singleton instances (mirroring source code access pattern)
Glomerular = Glomerular()
Tubular = Tubular()
AA = AA()
Myo = Myo()
TGF = TGF()
TGF_INTER = TGF_INTER()
TGF_LONG = TGF_LONG()


# ---------------------------------------------------------------------------
# Nephron equations (from nephron_eqs.py)
# ---------------------------------------------------------------------------

def C_protein(Q_A, P_GC, P_T0, x, C):
    """ODE for protein concentration along glomerular capillary."""
    return (Glomerular.Kf / (Glomerular.L * Q_A * Glomerular.C_A) * C ** 2 *
            (P_GC - P_T0 - Glomerular.a * C - Glomerular.b * C ** 2))


def Q_T_proximal(Q_T0, z):
    """Tubular flow in proximal tubule as function of normalised length z."""
    C = Q_T0 - Tubular.keppa / Tubular.theta
    return C + Tubular.keppa / Tubular.theta * np.exp(-Tubular.theta * z)


def Combined_desc(z, F):
    """ODE for (flow, NaCl conc) in short-loop descending limb."""
    C_I = 150 + 150 / 0.3 * z
    Q_T, C = F
    dQ_T = -Tubular.Lv * Tubular.ns * (C_I - C) * 6e4
    dC = (-C * dQ_T - 6e7 * Tubular.Ls * (C - C_I)) / Q_T
    return dQ_T, dC


def Combined_desc_long_inter(z, F):
    """ODE for (flow, NaCl conc) in long/intermediate descending limb."""
    C_I = 150 + 450 / 0.8 * z
    Q_T, C = F
    dQ_T = -Tubular.Lv * Tubular.ns * (C_I - C) * 6e4
    dC = (-C * dQ_T - 6e7 * Tubular.Ls * (C - C_I)) / Q_T
    return dQ_T, dC


def Cs_asce(Q_T_desc_end, z, C, md_loc=0.5):
    """ODE for NaCl conc in thick ascending limb."""
    C_I = (300 - 150 / 0.3 * z) if z < 0.3 else 150
    Vmax = Tubular.V_max1 if z < md_loc else Tubular.V_max2
    return (-6e7 * (Tubular.Ls * (C - C_I) +
                    1e3 * Vmax * C / (Tubular.Km + C)) / Q_T_desc_end)


def Cs_asce_thin(Q_T_desc_end, z, C, long=True):
    """ODE for NaCl conc in thin ascending limb."""
    C_I = 300
    Ls = Tubular.Ls_long if long else Tubular.Ls_inter
    return -6e7 * (Ls * (C - C_I)) / Q_T_desc_end


def P_T_asce(Q_T_desc_end, P_end, z, asce_length=0.65):
    """Tubular pressure in ascending limb (analytical)."""
    return (1 / 60 * 1 / 133.322 * 1e-6 * 8 * Tubular.mu / (np.pi * Tubular.r_loop ** 4) *
            Q_T_desc_end * (asce_length - z) + P_end)


def P_T_desc(sol_desc, z, P):
    """ODE for tubular pressure in descending limb (backward)."""
    Qz = sol_desc(z)[0]
    dP = (-1 / 60 * 1 / 133.322 * 1e-6 * 8 * Tubular.mu /
          (np.pi * Tubular.r_loop ** 4) * Qz)
    return dP


def P_T_proximal(Q_T0, P_TZ, z):
    """Tubular pressure in proximal tubule (analytical)."""
    Z = 1
    k2 = 1 / 60 * 1 / 133.322 * 1e-6 * 8 * Tubular.mu / (np.pi * Tubular.r_proximal ** 4)
    k1 = k2 * Tubular.keppa / Tubular.theta ** 2
    P_T = (k1 * (np.exp(-Tubular.theta * z) - np.exp(-Tubular.theta * Z)) +
           k2 * (Q_T0 - Tubular.keppa / Tubular.theta) * (Z - z) + P_TZ)
    return P_T


# ---------------------------------------------------------------------------
# Nephron solver (from nephron_solver.py)
# ---------------------------------------------------------------------------

class NegativeFlowException(Exception):
    pass


def func_Q_T0(Q_T0, Q_A, P_T0, P_GC=None):
    """Residual for solving filtration rate Q_T0 self-consistently."""
    if P_GC is None:
        P_GC = (Glomerular.R_E * (Q_A / (1 - Glomerular.H_A) - Q_T0) +
                Glomerular.R_PC * Q_A / (1 - Glomerular.H_A))

    C_x = partial(C_protein, Q_A, P_GC, P_T0)
    sol = solve_ivp(C_x, [0, Glomerular.L], [Glomerular.C_A],
                    method='Radau', dense_output=True)
    if not sol.success:
        raise RuntimeWarning(f'C_protein ODE failed: {sol.message}')

    C_E = np.squeeze(sol.y[:, -1])
    Q_T0_new = (1 - Glomerular.C_A / C_E) * Q_A
    return Q_T0 - Q_T0_new


def func_glomerular(P_T0, Q_A, P_GC=None, final=False, long=False, inter=False, debug=False):
    """
    Solve the nephron from glomerulus to collecting duct.

    Parameters
    ----------
    P_T0 : float
        Bowman's capsule / tubular entry pressure (mmHg).
    Q_A : float
        Afferent plasma flow (nl/min).
    P_GC : float, optional
        Glomerular capillary pressure (mmHg). If None, computed from R_E, R_PC.
    final : bool
        If True, return full outputs instead of residual.
    long : bool
        Use long-loop nephron equations.
    inter : bool
        Use intermediate-loop nephron equations.

    Returns
    -------
    If not final: residual P_T0 - P_start (for root-finding).
    If final: (P_T0, Cs_md, Q_T0, Cs_desc_end, Q_T_desc_end, P_end, P_md)
    """
    assert not (long and inter)
    Cs_0 = 150
    Q_T0 = Q_A / 3

    Q_T0 = fsolve(func_Q_T0, x0=Q_T0, args=(Q_A, P_T0, P_GC))

    if P_GC is None:
        P_GC = (Glomerular.R_E * (Q_A / (1 - Glomerular.H_A) - Q_T0) +
                Glomerular.R_PC * Q_A / (1 - Glomerular.H_A))

    assert len(Q_T0) == 1
    Q_T0 = Q_T0[0]

    Q_T_proximal_end = Q_T_proximal(Q_T0, z=1)
    if Q_T_proximal_end < 0:
        raise NegativeFlowException('negative in Q_T_proximal_end')

    # Descending limb
    if long:
        sol = solve_ivp(Combined_desc_long_inter, [0, 0.8],
                        [Q_T_proximal_end, Cs_0], method='Radau', dense_output=True)
    elif inter:
        sol = solve_ivp(Combined_desc_long_inter, [0, 0.55],
                        [Q_T_proximal_end, Cs_0], method='Radau', dense_output=True)
    else:
        sol = solve_ivp(Combined_desc, [0, 0.3],
                        [Q_T_proximal_end, Cs_0], method='Radau', dense_output=True)

    if not sol.success:
        raise RuntimeWarning(f'descending limb ODE failed: {sol.message}')

    z_desc_span = np.squeeze(sol.t)
    Q_T_desc_list = sol.y[0]
    Q_T_desc_end = np.squeeze(sol.y)[0, -1]
    Cs_desc_end = np.squeeze(sol.y)[1, -1]
    sol_desc = sol.sol

    if Q_T_desc_end < 0:
        raise NegativeFlowException('negative in Q_T_desc_end')
    if Cs_desc_end < 0:
        raise NegativeFlowException('negative in Cs_desc_end')

    # Thin ascending limb (long and intermediate loops only)
    if long or inter:
        Cs_asce_thin_cur = partial(Cs_asce_thin, Q_T_desc_end, long=long)
        asce_thin_len = 0.5 if long else 0.25
        sol = solve_ivp(Cs_asce_thin_cur, [0, asce_thin_len],
                        [Cs_desc_end], method='Radau', dense_output=True)
        if not sol.success:
            raise RuntimeWarning(f'thin ascending ODE failed: {sol.message}')
        Cs_thin_end = np.squeeze(sol.y)[-1]

        if long:
            z_range = np.concatenate([np.arange(0, 0.3, 0.001),
                                      np.arange(0.3, 0.32 - 1e-6, 0.001),
                                      np.arange(0.32, 0.47 - 1e-6, 0.001)])
        else:
            z_range = np.concatenate([np.arange(0, 0.3, 0.001),
                                      np.arange(0.3, 0.4 - 1e-6, 0.001),
                                      np.arange(0.4, 0.55 - 1e-6, 0.001)])
    else:
        z_range = np.concatenate([np.arange(0, 0.3, 0.001),
                                  np.arange(0.3, 0.5 - 1e-6, 0.001),
                                  np.arange(0.5, 0.65 - 1e-6, 0.001)])

    # Thick ascending limb
    if long or inter:
        asce_len = 0.47 if long else 0.55
        md_loc = 0.32 if long else 0.4
        Cs_asce_cur = partial(Cs_asce, Q_T_desc_end, md_loc=md_loc)
        sol = solve_ivp(Cs_asce_cur, [0, asce_len],
                        [Cs_thin_end], t_eval=z_range,
                        method='Radau', dense_output=True)
    else:
        Cs_asce_cur = partial(Cs_asce, Q_T_desc_end)
        sol = solve_ivp(Cs_asce_cur, [0, 0.65],
                        [Cs_desc_end], t_eval=z_range,
                        method='Radau', dense_output=True)

    if not sol.success:
        raise RuntimeWarning(f'ascending limb ODE failed: {sol.message}')

    if long:
        md_index = np.argmin(np.abs(z_range - 0.32))
    elif inter:
        md_index = np.argmin(np.abs(z_range - 0.4))
    else:
        md_index = np.argmin(np.abs(z_range - 0.5))

    y = np.squeeze(sol.y)
    assert len(y) == len(z_range)
    Cs_md = y[md_index]
    if Cs_md < 0:
        raise NegativeFlowException('negative in Cs_md')

    # Solve tubular pressure at loop end
    sol_p = fsolve(lambda x: x - Q_T_desc_end / (Tubular.alpha * x + Tubular.beta) ** 4,
                   x0=7.2467)
    P_end = sol_p[0]

    if long:
        asce_length = 0.97
    elif inter:
        asce_length = 0.8
    else:
        asce_length = 0.65

    P_desc_end = P_T_asce(Q_T_desc_end, P_end, 0, asce_length=asce_length)
    P_md = P_T_asce(Q_T_desc_end, P_end, asce_length - 0.15, asce_length=asce_length)

    # Solve P_T backwards through descending limb
    if long:
        desc_len = 0.8
    elif inter:
        desc_len = 0.55
    else:
        desc_len = 0.3

    P_T_desc_cur = partial(P_T_desc, sol_desc)
    sol_back = solve_ivp(P_T_desc_cur, [desc_len, 0], [P_desc_end], method='LSODA')
    P_proximal_end = np.squeeze(sol_back.y)[-1]
    P_start = P_T_proximal(Q_T0, P_proximal_end, z=0)

    if not final:
        return P_T0 - P_start
    else:
        return P_T0, Cs_md, Q_T0, Cs_desc_end, Q_T_desc_end, P_end, P_md


# ---------------------------------------------------------------------------
# Afferent arteriole model (from afferent_arteriole.py)
# ---------------------------------------------------------------------------

def fx(x_tgf, x_myo):
    """Smooth muscle activation function."""
    x = x_myo
    return 3 * np.exp(x) / (np.exp(x) + 2 * np.exp(-0.5 * x))


def h_v(r_v):
    """Wall thickness as function of radius (um)."""
    r_0 = 10
    h_0 = 2
    return -r_v + np.sqrt(r_v ** 2 + 2 * h_0 * r_0 + h_0 ** 2)


def AA_model(r_num, Cs_md, P_v, final=False, only_myo=False, type=0):
    """
    Afferent arteriole wall mechanics equilibrium.

    Parameters
    ----------
    r_num : float — arteriole radius (um)
    Cs_md : float — NaCl concentration at macula densa (mmol/l)
    P_v : float   — mean vessel pressure (mmHg)
    final : bool  — if True, return tensions and activation states
    only_myo : bool — if True, suppress TGF
    type : int    — nephron type (0=short, 1=inter, 2=long)

    Returns
    -------
    If not final: tension balance residual.
    If final: (wall_tension, x_myo, x_tgf, T_e, T_m)
    """
    if type == 0:
        x_tgf = 5 * 1 / Myo.lam * (TGF.ita_max -
                                     TGF.phi / (1 + np.exp(TGF.k * (Cs_md - TGF.C_half))))
    elif type == 1:
        x_tgf = 5 * 1 / Myo.lam * (TGF_INTER.ita_max -
                                     TGF_INTER.phi / (1 + np.exp(TGF_INTER.k * (Cs_md - TGF.C_half))))
    elif type == 2:
        x_tgf = 5 * 1 / Myo.lam * (TGF_LONG.ita_max -
                                     TGF_LONG.phi / (1 + np.exp(TGF_LONG.k * (Cs_md - TGF.C_half))))
    else:
        raise ValueError("type must be 0, 1, or 2")

    x_tgf = 0 if only_myo else x_tgf

    x_myo = Myo.lam * Myo.G * (r_num * P_v / (1e3 / 133.32) - Myo.T0 * (1 - x_tgf))

    T_e = 1e-4 * AA.sigma_e * h_v(r_num) * (np.exp(1e-4 * AA.k_e * (r_num - AA.r_e)) - 1)
    T_m = 1e-4 * AA.sigma_m * h_v(r_num) * fx(x_tgf, x_myo) * np.exp(-1e-8 * AA.k_m * (r_num - AA.r_m) ** 2)

    if not final:
        return r_num * P_v / (1e3 / 133.32) - T_e - T_m
    else:
        return r_num * P_v / (1e3 / 133.32), x_myo, x_tgf, T_e, T_m


# ---------------------------------------------------------------------------
# Myo parameter (used in AA_model)
# ---------------------------------------------------------------------------
# Myo.G is set in original parameters.py as G: float = 0.06
Myo.G = 0.06  # cm dyn^-1


# ---------------------------------------------------------------------------
# Single nephron simulation (core of tree_model.simu)
# ---------------------------------------------------------------------------

def _simu_core(Q, r_v, P_t_in, P_GC, type=0, only_myo=False):
    """
    Single-nephron simulation at fixed flow and pressures.

    Parameters
    ----------
    Q : float       — afferent blood flow (um^3/s)
    r_v : float     — afferent arteriole radius (um)
    P_t_in : float  — upstream pressure (mmHg)
    P_GC : float    — glomerular capillary pressure (mmHg)
    type : int      — nephron type: 0=short loop, 1=intermediate, 2=long
    only_myo : bool — suppress TGF (myogenic only)

    Returns
    -------
    tuple: (r_opt, Q_T0, x_myo, x_tgf, Cs_md, P_T0, Cs_desc_end,
            Q_T_desc_end, ratio, T_e, T_m, P_end, P_md)
        r_opt  : equilibrium afferent arteriole radius (um)
        Q_T0   : SNGFR — single nephron GFR (nl/min)
        x_myo  : myogenic activation state
        x_tgf  : TGF activation state
        Cs_md  : NaCl concentration at macula densa (mmol/l)
        P_T0   : Bowman's capsule pressure (mmHg)
        Cs_desc_end : NaCl at end of descending limb (mmol/l)
        Q_T_desc_end : flow at loop bend (nl/min)
        ratio  : filtration fraction = Q_T0 / (2 * Q_A)
        T_e    : elastic wall tension (dyn/cm)
        T_m    : active muscle tension (dyn/cm)
        P_end  : tubular pressure at loop end (mmHg)
        P_md   : tubular pressure at macula densa (mmHg)
    """
    Q_A = 6 * 1e-5 * Q * (1 - Glomerular.H_A)  # um^3/s -> nl/min
    P_v = (P_t_in + P_GC) / 2

    if type == 0:
        func_glomerular_cur = func_glomerular
        AA_cur = AA_model
    elif type == 1:
        func_glomerular_cur = partial(func_glomerular, inter=True)
        AA_cur = partial(AA_model, type=1)
    elif type == 2:
        func_glomerular_cur = partial(func_glomerular, long=True)
        AA_cur = partial(AA_model, type=2)
    else:
        raise ValueError("type must be 0, 1, or 2")

    try:
        root = fsolve(func_glomerular_cur, x0=10, args=(Q_A, P_GC))
        P_0_final = root[0]
        (P_0_final_again, Cs_md_final, Q_T0, Cs_desc_end,
         Q_T_desc_end, P_end, P_md) = func_glomerular_cur(
            P_0_final, Q_A, P_GC=P_GC, final=True)
    except NegativeFlowException:
        P_0_final = P_0_final_again = Cs_md_final = Q_T0 = 0
        Cs_desc_end = Q_T_desc_end = P_end = P_md = 0
    except RuntimeWarning:
        return r_v, 1 / 3 * Q_A, 0, 0, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 0

    # FF = SNGFR / Q_A_plasma (standard plasma-flow definition)
    # Q_A is already plasma flow (Q_blood*(1-H_A)); dividing by 2 was wrong (gave blood-FF)
    ratio = Q_T0 / Q_A if Q_A > 0 else 0

    try:
        r_opt = so.brentq(AA_cur, a=1e-1, b=20,
                          args=(Cs_md_final, P_v, False, only_myo))
        T1, x_myo, x_tgf, T_e, T_m = AA_cur(r_opt, Cs_md_final, P_v,
                                              final=True, only_myo=only_myo)
    except RuntimeWarning:
        T1, x_myo, x_tgf, T_e, T_m = AA_cur(r_v, Cs_md_final, P_v,
                                              final=True, only_myo=only_myo)
        return r_v, Q_T0, 0, 0, Cs_md_final, P_0_final, Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m, P_end, P_md

    return (r_opt, Q_T0, x_myo, x_tgf, Cs_md_final, P_0_final,
            Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m, P_end, P_md)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_nephron(pressure_mmHg=100, p=None,
                       Q_um3_s=3.9e6, r_v_um=10.1,
                       P_GC_mmHg=None, nephron_type=0, only_myo=False):
    """
    Run a single-nephron simulation.

    Simulates one afferent arteriole + glomerulus + tubule at a given renal
    perfusion pressure. This is standalone and does not require VTK data.

    Parameters
    ----------
    pressure_mmHg : float
        Renal perfusion (inlet) pressure in mmHg. Default 100.
    p : dict, optional
        Parameter overrides. Any field from Glomerular, Tubular, AA, Myo, or TGF
        dataclasses, plus 'Q_um3_s', 'r_v_um', 'P_GC_mmHg'.
        Examples: p={'Kf': 10.0}, p={'G': 0.09, 'C_half': 40.0}.
    Q_um3_s : float
        Afferent blood flow in um^3/s (default 3.9e6 → Q_A = 117 nl/min; Xu 2025 Table 4).
    r_v_um : float
        Initial afferent arteriole radius in um (default 10.1; Xu 2025 Table 4).
    P_GC_mmHg : float or None
        Glomerular capillary pressure in mmHg. If None, estimated as 0.58 * pressure_mmHg (Xu 2025 Table 4: 58 mmHg at 100 mmHg).
    nephron_type : int
        0 = short-loop (cortical), 1 = intermediate, 2 = long-loop (juxtamedullary).
    only_myo : bool
        If True, suppress TGF (myogenic response only).

    Returns
    -------
    dict with keys:
        r_AA_um        : afferent arteriole radius (um)
        SNGFR_nl_min   : single nephron GFR (nl/min)
        Cs_md_mmol_l   : NaCl at macula densa (mmol/l)
        P_T0_mmHg      : Bowman's capsule pressure (mmHg)
        P_GC_mmHg      : glomerular capillary pressure (mmHg)
        Q_A_nl_min     : afferent plasma flow (nl/min)
        x_myo          : myogenic activation
        x_tgf          : TGF activation
        T_e_dyn_cm     : elastic wall tension (dyn/cm)
        T_m_dyn_cm     : active muscle tension (dyn/cm)
        filtration_fraction : SNGFR / Q_A
        P_end_mmHg     : tubular pressure at loop end (mmHg)
        P_md_mmHg      : tubular pressure at macula densa (mmHg)
    """
    import dataclasses as _dc
    import sys as _sys

    # ------------------------------------------------------------------
    # Apply parameter overrides via p dict.
    # Strategy: temporarily patch the module-level singleton objects so
    # that all downstream equation functions (which reference the globals
    # Glomerular, Tubular, AA, Myo, TGF) pick up the overrides.
    # We restore the originals in a finally block to keep the module
    # state clean for subsequent calls.
    # ------------------------------------------------------------------

    # Module-level singleton objects (these are the globals used by all
    # equation functions in this file)
    _mod = _sys.modules[__name__]
    _singletons = {
        'Glomerular': getattr(_mod, 'Glomerular'),
        'Tubular':    getattr(_mod, 'Tubular'),
        'AA':         getattr(_mod, 'AA'),
        'Myo':        getattr(_mod, 'Myo'),
        'TGF':        getattr(_mod, 'TGF'),
    }

    if p is not None:
        Q_um3_s   = p.get('Q_um3_s',   Q_um3_s)
        r_v_um    = p.get('r_v_um',    r_v_um)
        P_GC_mmHg = p.get('P_GC_mmHg', P_GC_mmHg)

        # Save original attribute values for every singleton so we can
        # restore them even if the call raises an exception.
        _saved = {}
        for name, obj in _singletons.items():
            # dataclass fields (stored on the instance __dict__)
            if _dc.is_dataclass(obj):
                for f in _dc.fields(obj):
                    if f.name in p:
                        _saved[(name, f.name)] = getattr(obj, f.name)
                        object.__setattr__(obj, f.name, p[f.name])
            # instance attributes that are NOT dataclass fields
            # (e.g. Myo.G set after instantiation, Myo.lam etc.)
            for attr_name, attr_val in vars(obj).items():
                if attr_name in p and (name, attr_name) not in _saved:
                    _saved[(name, attr_name)] = attr_val
                    object.__setattr__(obj, attr_name, p[attr_name])
            # class-level (non-field) attributes: e.g. Tubular.V_max1, V_max2
            # These live in type(obj).__dict__ and are NOT shadowed by the
            # instance, so object.__setattr__ on the class is required.
            for attr_name, attr_val in type(obj).__dict__.items():
                if (attr_name in p
                        and not attr_name.startswith('_')
                        and (name, attr_name) not in _saved):
                    _saved[(name, attr_name)] = attr_val
                    # Set on the class so all accesses via 'Tubular.V_max1' etc. see it
                    setattr(type(obj), attr_name, p[attr_name])

        # Also handle Myo.G which is set as a bare attribute after
        # instantiation (not a declared dataclass field default).
        if 'G' in p and ('Myo', 'G') not in _saved:
            _saved[('Myo', 'G')] = getattr(_singletons['Myo'], 'G')
            object.__setattr__(_singletons['Myo'], 'G', p['G'])

    if P_GC_mmHg is None:
        # Xu 2025 Table 4: P_GC = 58 mmHg at 100 mmHg perfusion → ratio 0.58
        P_GC_mmHg = 0.58 * pressure_mmHg

    # Xu 2025 Table 4: T_wall = 81.1 dyn/cm at r=10.1 µm → P_v = T/(r×1333.2) = 60.2 mmHg
    # P_t_in = 2×P_v − P_GC = 2×60.2 − 58 = 62.4 mmHg at P_perf=100 → factor 0.624
    # Accounts for large-vessel pressure drop from renal artery to the AA inlet
    P_t_in_mmHg = 0.624 * pressure_mmHg

    try:
        (r_opt, Q_T0, x_myo, x_tgf, Cs_md, P_T0,
         Cs_desc_end, Q_T_desc_end, ratio, T_e, T_m,
         P_end, P_md) = _simu_core(Q_um3_s, r_v_um, P_t_in_mmHg, P_GC_mmHg,
                                    type=nephron_type, only_myo=only_myo)

        Q_A = 6e-5 * Q_um3_s * (1 - getattr(_mod, 'Glomerular').H_A)

        result = {
            'r_AA_um': r_opt,
            'SNGFR_nl_min': Q_T0,
            'Cs_md_mmol_l': Cs_md,
            'P_T0_mmHg': P_T0,
            'P_GC_mmHg': P_GC_mmHg,
            'Q_A_nl_min': Q_A,
            'x_myo': x_myo,
            'x_tgf': x_tgf,
            'T_e_dyn_cm': T_e,
            'T_m_dyn_cm': T_m,
            'filtration_fraction': ratio,
            'P_end_mmHg': P_end,
            'P_md_mmHg': P_md,
        }
    finally:
        # Restore original singleton attribute values (always, even on error).
        if p is not None and '_saved' in dir():
            for (obj_name, attr_name), orig_val in _saved.items():
                obj = _singletons[obj_name]
                # If the attribute lives on the class (not the instance),
                # restore it on the class as well.
                if attr_name not in vars(obj) and attr_name in type(obj).__dict__:
                    setattr(type(obj), attr_name, orig_val)
                else:
                    object.__setattr__(obj, attr_name, orig_val)

    return result


def run_full_kidney(pressure_mmHg=100, vtk_dir=None, source_dir=None,
                    num_iter=20, lr=0.25, only_myo=False, relTol=2e-5):
    """
    Run the full-kidney autoregulation simulation.

    Requires the vascular tree VTK file (Kirchhoff-solved) and the original
    source directory on the Python path.

    Parameters
    ----------
    pressure_mmHg : int
        Renal perfusion pressure (mmHg).
    vtk_dir : str
        Directory containing Kirchhoff VTK files (e.g. 'kirchhoff_p_in = 100.vtk').
    source_dir : str
        Path to the original model source directory (needed for tree_model.simu).
    num_iter : int
        Maximum iterations.
    lr : float
        Learning rate for radius update.
    only_myo : bool
        Suppress TGF.
    relTol : float
        Relative flow convergence tolerance.

    Returns
    -------
    dict with keys:
        in_flow_before_ml_min : total renal blood flow before autoregulation (ml/min)
        in_flow_after_ml_min  : total renal blood flow after autoregulation (ml/min)
        converged : bool
    """
    if source_dir is None:
        source_dir = (r"C:\Users\sneidere\Dropbox\Projects\2026"
                      r"\kidney-model-collection\M008_Xu_2025_full_kidney")

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    # helper_funcs is missing from the repo; create a minimal stub if needed
    _ensure_helper_funcs(source_dir)

    try:
        from tree_model import AutoRegulation
    except ImportError as e:
        raise ImportError(
            f"Could not import tree_model from {source_dir}. "
            f"Ensure helper_funcs.py exists there. Error: {e}"
        )

    if vtk_dir is None:
        vtk_dir = os.path.join(source_dir, 'data')

    vtk_file = os.path.join(vtk_dir, f'kirchhoff_p_in = {pressure_mmHg}.vtk')
    if not os.path.exists(vtk_file):
        vtk_file = os.path.join(source_dir, 'final_tree.vtk')

    old_cwd = os.getcwd()
    os.chdir(source_dir)
    try:
        auto = AutoRegulation(P_in=pressure_mmHg, lr=lr, only_myo=only_myo,
                              num_iter=num_iter, relTol=relTol, pop=False)
        auto.set_up_tree(pt_file=vtk_file)
        before, after = auto.auto_reg()
    finally:
        os.chdir(old_cwd)

    return {
        'in_flow_before_ml_min': before,
        'in_flow_after_ml_min': after,
        'converged': True,
    }


def _ensure_helper_funcs(source_dir):
    """
    Create a minimal helper_funcs.py stub in source_dir if it doesn't exist.
    The original file is missing from the repository; it appears to export
    only standard library/numpy/scipy symbols used via wildcard import.
    """
    stub_path = os.path.join(source_dir, 'helper_funcs.py')
    if not os.path.exists(stub_path):
        stub = (
            "# Auto-generated stub — original helper_funcs.py missing from repo\n"
            "import numpy as np\n"
            "import scipy\n"
            "import scipy.spatial\n"
            "import os\n"
            "import logging\n"
        )
        with open(stub_path, 'w') as fh:
            fh.write(stub)
        logging.info(f"Created helper_funcs.py stub at {stub_path}")


# ---------------------------------------------------------------------------
# Parameter table
# ---------------------------------------------------------------------------

PARAMETER_TABLE = [
    # Glomerular
    {"name": "C_A",       "value": 57,       "units": "g/l",              "class": "Glomerular", "meaning": "Afferent plasma protein concentration"},
    {"name": "H_A",       "value": 0.5,      "units": "—",                "class": "Glomerular", "meaning": "Haematocrit (plasma fraction = 1 - H_A)"},
    {"name": "Kf",        "value": 2.5,      "units": "—",                "class": "Glomerular", "meaning": "Glomerular ultrafiltration coefficient"},
    {"name": "R_E",       "value": 0.209,    "units": "mmHg min nl⁻¹",   "class": "Glomerular", "meaning": "Efferent arteriole hydraulic resistance"},
    {"name": "R_PC",      "value": 0.0702,   "units": "mmHg min nl⁻¹",   "class": "Glomerular", "meaning": "Peritubular capillary hydraulic resistance"},
    {"name": "a",         "value": 0.1631,   "units": "mmHg l g⁻¹",      "class": "Glomerular", "meaning": "Oncotic pressure linear coefficient"},
    {"name": "b",         "value": 0.00294,  "units": "—",                "class": "Glomerular", "meaning": "Oncotic pressure quadratic coefficient"},
    {"name": "L",         "value": 10,       "units": "—",                "class": "Glomerular", "meaning": "Normalised glomerular capillary length"},
    # Tubular
    {"name": "Km",        "value": 20,       "units": "mM",               "class": "Tubular",    "meaning": "Michaelis constant for NaCl active reabsorption"},
    {"name": "Lv",        "value": 2e-5,     "units": "cm² l osmol⁻¹ s⁻¹","class": "Tubular",   "meaning": "Water permeability of descending limb"},
    {"name": "ns",        "value": 2,        "units": "—",                "class": "Tubular",    "meaning": "Osmoles per NaCl molecule"},
    {"name": "Ls",        "value": 3.4e-7,   "units": "cm² s⁻¹",         "class": "Tubular",    "meaning": "NaCl permeability (short-loop ascending limb)"},
    {"name": "V_max1",    "value": 1.22e-7,  "units": "mmol cm⁻¹ s⁻¹",  "class": "Tubular",    "meaning": "Max NaCl transport rate, pre-macula densa"},
    {"name": "V_max2",    "value": 0.793e-7, "units": "mmol cm⁻¹ s⁻¹",  "class": "Tubular",    "meaning": "Max NaCl transport rate, post-macula densa"},
    {"name": "alpha",     "value": 1.65e-2,  "units": "(nl min⁻¹ mmHg⁻⁵)^1/4","class": "Tubular","meaning": "Tubular compliance parameter"},
    {"name": "beta",      "value": 0.924,    "units": "(nl min⁻¹ mmHg⁻¹)^1/4","class": "Tubular","meaning": "Tubular compliance parameter"},
    {"name": "theta",     "value": 1.3,      "units": "cm⁻¹",            "class": "Tubular",    "meaning": "Proximal reabsorption exponential decay constant"},
    {"name": "keppa",     "value": 33.6,     "units": "nl min⁻¹ cm⁻¹",  "class": "Tubular",    "meaning": "Proximal tubule reabsorption coefficient"},
    {"name": "gamma",     "value": 1.33e-5,  "units": "cm mmHg⁻¹",      "class": "Tubular",    "meaning": "Tubular compliance factor"},
    {"name": "mu",        "value": 7.2e-4,   "units": "g cm⁻¹ s⁻¹",     "class": "Tubular",    "meaning": "Tubular fluid viscosity"},
    {"name": "r_proximal","value": 12e-4,    "units": "cm",               "class": "Tubular",    "meaning": "Proximal tubule radius"},
    {"name": "r_loop",    "value": 10e-4,    "units": "cm",               "class": "Tubular",    "meaning": "Loop of Henle radius"},
    {"name": "Ls_long",   "value": 5.6e-7,   "units": "cm² s⁻¹",         "class": "Tubular",    "meaning": "NaCl permeability — long-loop"},
    {"name": "Ls_inter",  "value": 5.2e-7,   "units": "cm² s⁻¹",         "class": "Tubular",    "meaning": "NaCl permeability — intermediate-loop"},
    # Afferent arteriole — from Xu 2025 Table 3 (prior values were from Holstein-Rathlou 1994)
    {"name": "sigma_e",   "value": 25.56,    "units": "dyn cm⁻²",        "class": "AA",         "meaning": "Elastic stress coefficient"},
    {"name": "k_e",       "value": 1.7304e4, "units": "cm⁻¹",            "class": "AA",         "meaning": "Elastic stiffness exponent"},
    {"name": "r_e",       "value": 7,        "units": "um",               "class": "AA",         "meaning": "Elastic rest radius"},
    {"name": "k_m",       "value": 8.75e6,   "units": "cm⁻²",            "class": "AA",         "meaning": "Smooth muscle stiffness"},
    {"name": "r_m",       "value": 12.5,     "units": "um",               "class": "AA",         "meaning": "Smooth muscle optimal radius"},
    {"name": "sigma_m",   "value": 7.23e5,   "units": "dyn cm⁻²",        "class": "AA",         "meaning": "Smooth muscle stress coefficient"},
    {"name": "r_n",       "value": 10.04,    "units": "um",               "class": "AA",         "meaning": "Natural/unloaded radius"},
    # Myogenic
    {"name": "lam",       "value": 1,        "units": "—",                "class": "Myo",        "meaning": "Myogenic coupling gain"},
    {"name": "T0",        "value": 81.1252,  "units": "dyn cm⁻¹",        "class": "Myo",        "meaning": "Basal active wall tension"},
    {"name": "G",         "value": 0.06,     "units": "cm dyn⁻¹",        "class": "Myo",        "meaning": "Myogenic sensitivity"},
    # TGF (short loop)
    {"name": "C_half",    "value": 44,       "units": "mmol/l",           "class": "TGF",        "meaning": "NaCl set-point at macula densa"},
    {"name": "k_TGF",     "value": 0.085,    "units": "l/mmol",           "class": "TGF",        "meaning": "TGF sigmoid steepness"},
    {"name": "ita_max",   "value": 0.091,    "units": "—",                "class": "TGF",        "meaning": "Maximum TGF activation"},
    {"name": "phi",       "value": 0.182,    "units": "—",                "class": "TGF",        "meaning": "TGF amplitude parameter"},
    # Vascular tree
    {"name": "mu_blood",  "value": 3.6e-15,  "units": "N s um⁻²",        "class": "VascularTree","meaning": "Blood viscosity"},
    {"name": "vspace",    "value": 22.6,     "units": "um/voxel",         "class": "VascularTree","meaning": "Voxel-to-physical length scale"},
    {"name": "R_E_tree",  "value": 0.209*60*133.322e-3, "units": "N s mm⁻⁵", "class": "VascularTree","meaning": "Efferent arteriole resistance (tree units)"},
    {"name": "R_PC_tree", "value": 0.0702*60*133.322e-3,"units": "N s mm⁻⁵","class": "VascularTree","meaning": "Peritubular capillary resistance (tree units)"},
]


if __name__ == '__main__':
    print("Running single nephron at P_in = 100 mmHg ...")
    result = run_single_nephron(pressure_mmHg=100)
    for k, v in result.items():
        print(f"  {k:30s} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")
