"""
M002 Richfield 2024 — Glomerular Autoregulation Model
Python port of Richfield et al. 2024 R implementation.

Reference: Richfield et al. 2024, glomerular autoregulation model combining
myogenic + TGF mechanisms with capillary network filtration.

Key physics:
- 194-edge capillary network (Shea anatomy) solved iteratively
- Starling filtration law with oncotic pressure feedback
- Hematocrit-dependent viscosity (Pries empirical formula)
- Iterative convergence on Rinf (filtration resistance) and viscosity

Reference baseline (Pa=100 mmHg, D=7 um, Ca=5.94 g/dl):
  SNGFR     = 29.71 nl/min
  Pgc_mean  = 49.88 mmHg  (capillary average)
  Pgc_inlet = 51.18 mmHg  (pressure at afferent exit / inlet to glom)
  FF        = 0.292
"""

import numpy as np
from scipy.linalg import solve

# ============================================================
# Parameters (from parms.R)
# ============================================================
params = {
    # Pressure boundary conditions (mmHg)
    "Paa_i": 100.0,       # afferent arteriole inlet pressure
    "Pea_o": 15.0,        # efferent arteriole outlet pressure
    "P_bs":  13.0,        # Bowman's space pressure
    "P_ext": 5.0,         # external tissue pressure

    # Arteriole geometry (um)
    "DAA_0": 7.0,         # baseline afferent arteriole diameter
    "DAA_pass": 22.0,     # passive diameter
    "L_AA": 106.6698,     # afferent arteriole length (um)
    "DEA_0": 7.3,         # efferent arteriole diameter
    "L_EA": 106.6698,     # efferent arteriole length (um)

    # Glomerular parameters
    "k": 3e-5,            # hydraulic conductivity (nl/min/mmHg/um^2)
    "Ca_i": 5.93868,      # plasma protein concentration (g/dl)
    "t": 138 + 40,        # capillary wall thickness (nm)
    "hpod": 300,          # podocyte foot process height (nm)
    "wpod": 0.17,         # podocyte foot process width
    "cmp_0": 0.02081642,
    "YM": 14.36735,
    "YMn": 15.8405,
    "V_0": 512371.7,
    "V_0n": 559191.6,

    # Systemic parameters
    "mu_plas": 1.24,      # plasma viscosity (cP)
    "H_t_sys": 0.40,      # systemic hematocrit

    # Pre-afferent resistance (mmHg*min/nl * 60)
    "Rra": 1.09,          # nl*mmHg/s -> will be divided by 60 in R code

    # Solver parameters
    "num_iter": 150,
    "mu_tol": 1e-3,
    "Rinf_tol": 1e-3,
    "beta": 3,
}

# ============================================================
# Shea Glomerular Anatomy (from shea_anatomy_pressure.R)
# ============================================================
# Node numbering: in.nodes=1, out.nodes=195
# Source and target nodes for each edge

SOURCE_NODES = np.array([
    1, 2, 2, 3, 3, 4, 4, 4, 5, 5,
    6, 6, 7, 7, 7, 8, 8,
    9, 9, 10,
    10, 10, 11, 11, 11, 12, 13, 13, 14, 14,
    14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
    26, 27, 27, 28,
    28, 28, 29, 30, 31, 32,
    32, 32, 33, 33, 34, 35, 35, 36, 36, 37,
    37, 39, 40, 41, 42, 42, 44, 44, 45, 46,
    47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
    53, 53, 54, 55, 56, 56, 57, 58, 59, 59,
    59, 59, 59, 60, 60, 61, 61, 62, 62, 136,
    63, 63, 64, 65, 65, 66, 67, 68, 69, 69,
    69, 70, 70,
    71, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 79, 80, 81, 82, 82, 83, 83,
    84, 84, 85, 85, 86, 86, 87, 87, 88, 89,
    90, 90, 91, 91, 92, 92, 93, 93, 93, 94,
    94, 95, 96, 98, 98, 99, 99, 100, 100, 101,
    101, 102, 102, 102, 103, 103, 103, 104, 105, 105,
    106,
    106, 107, 107,
    108, 109, 109, 111, 110, 110,
    111, 112, 113, 114,
    116, 116, 117, 117, 118, 118,
    119, 120, 121, 122, 122, 123, 124, 125, 125, 126,
    126, 127, 128, 128, 129, 129, 130, 131, 131, 132,
    133, 134, 135, 135, 137, 139, 139, 140, 140, 141,
    141, 142, 142, 143, 143, 144, 144, 144, 144, 145,
    146, 146, 147, 147, 148, 148, 149, 149, 150, 151,
    151, 152, 152, 153, 153, 154, 154, 154, 155, 155,
    156, 156, 157, 157, 157, 158, 159, 160, 160, 161,
    161, 162, 162, 163, 165, 166, 166, 167, 167, 168,
    169, 170, 170, 171, 171, 172, 172, 172, 173, 174,
    175, 175, 177, 177, 178, 178, 179, 174, 180, 180,
    181, 182, 182,
    183, 184, 184, 184, 184, 185, 186,
    187, 188, 189, 189, 190, 191, 191, 191, 192, 193
], dtype=int)

TARGET_NODES = np.array([
    2, 3, 82, 4, 139, 5, 6, 7, 8, 9,
    12, 25, 10, 11, 29, 12, 19,
    13, 14, 15,
    16, 21, 17, 18, 23, 19, 31, 32, 42, 67,
    121, 20, 39, 21, 26, 22, 23, 24, 40, 25,
    26, 27, 28, 34, 30, 51, 29, 28, 30, 31,
    32, 32, 33, 34,
    43, 47, 35, 41, 36, 38,
    43, 45, 38, 46, 39, 40, 41, 37, 42, 38,
    45, 43, 44, 49, 45, 46, 47, 49, 48, 54,
    59, 50, 58, 52, 53, 54, 55, 56, 55, 56,
    57, 58, 65, 63, 59, 64, 60, 66, 61, 63,
    64, 69, 80, 65, 66, 67, 72, 68, 73, 62,
    68, 69, 74, 70, 75, 71, 72, 76, 73, 74,
    76, 75, 77,
    77, 77, 77, 76, 78, 77, 78,
    79, 81, 80, 81, 121, 194, 83, 84, 85, 86,
    87, 88, 88, 89, 90, 91, 89, 92, 94, 176,
    93, 96, 94, 95, 95, 97, 96, 98, 99, 97,
    97, 100, 108, 101, 102, 103, 119, 104, 105, 106,
    115, 107, 120, 123, 104, 108, 112, 113, 109, 111,
    115,
    116, 112, 113,
    117, 115, 118, 110, 114, 114,
    119, 124, 121, 119,
    118, 120, 122, 176, 123, 127,
    124, 130, 125, 126, 133, 127, 128, 129, 136, 132,
    193, 130, 131, 138, 132, 133, 134, 135, 137, 150,
    134, 136, 138, 138, 138, 140, 141, 142, 143, 144,
    145, 152, 159, 146, 168, 147, 164, 165, 148, 164,
    137, 149, 169, 174, 150, 151, 152, 153, 163, 154,
    155, 156, 157, 162, 158, 159, 163, 164, 160, 161,
    171, 173, 158, 162, 181, 135, 167, 164, 165, 163,
    166, 167, 175, 177, 168, 168, 170, 169, 169, 170,
    172, 174, 188, 172, 176, 173, 176, 179, 176, 177,
    178, 184, 179, 180, 181, 182, 183, 191, 183, 185,
    188, 184, 189,
    185, 186, 186, 187, 187, 193, 189,
    192, 190, 190, 191, 193, 192, 193, 193, 193, 194
], dtype=int)

EDGE_LENGTHS = np.array([
    3.8, 9.5, 11.5, 3.8, 7.9, 17.4, 16.1, 28.7, 5.1, 33,
    25.8, 10.7, 35.1, 5.4, 50.4, 14.8, 14.6,
    9.2, 41.4, 28.1,
    6.9, 20.2, 9.0, 26.4, 9.0, 13.1, 11.3, 20.0, 18.2, 2.5,
    11.4, 3.8, 33.3, 24.3, 25.6, 64.0, 10.2, 30.7, 53.2, 14.1,
    16.4, 6.9, 22.5, 3.8, 19.2, 23.0, 39.1, 21.7, 18.2, 10.2,
    11.0, 24.3, 11.5, 12.8,
    14.3, 23.0, 16.9, 11.3, 25.6, 21.7,
    7.9, 24.3, 9.2, 34.5, 16.9, 24.3, 38.9, 21.5, 37.6, 34.3,
    21.5, 27.6, 12.3, 17.4, 7.7, 2.6, 13.6, 20.7, 9.5, 71.1,
    35.3, 10.5, 35.3, 24.6, 9.2, 11.5, 10.2, 33.8, 25.8, 24.3,
    10.7, 28.4, 27.9, 15.3, 25.6, 8.2, 33.5, 21.7, 8.9, 11.8,
    23.5, 20.5, 24.0, 7.7, 6.9, 24.6, 12.3, 28.1, 20.5, 20.0,
    6.9, 15.6, 21.0, 12.8, 24.3, 13.8, 20.5, 28.9, 30.2, 15.3,
    26.3, 4.4, 17.4, 30.7, 3.3, 26.9, 21.7, 16.9, 19.2, 5.4,
    11.5, 7.7, 10.7, 26.6, 32.2, 7.7, 27.1, 23.8, 37.1, 2.6,
    34.0, 27.9, 40.9, 13.1, 6.4, 15.6, 50.1, 49.1, 13.6, 14.3,
    80.8, 39.7, 18.4, 4.9, 23.0, 19.2, 25.6, 35.6, 28.1, 21.7,
    29.9, 58.8, 25.6, 8.2, 16.6, 14.1, 26.6, 37.6, 30.2, 7.4,
    18.4, 10.2, 5.1, 2.6, 16.1, 7.9, 15.9, 39.4, 109.0, 12.0,
    4.1, 23.8, 12.5, 55.5, 12.8, 7.7, 22.5, 23.0, 6.1, 6.7,
    49.9, 10.2, 9.5, 13.1, 8.7, 7.7, 17.9, 6.1, 9.2, 11.0,
    23.5, 28.4, 11.8, 15.3, 8.7, 23.3, 23.3, 18.4, 7.7, 30.7,
    9.0, 33.5, 11.5, 14.6, 17.9, 9.2, 21.7, 12.8, 33.0, 2.5,
    9.2, 6.4, 13.4, 29.2, 9.7, 17.4, 33.5, 20.0, 6.4, 37.1,
    25.6, 7.7, 38.4, 12.3, 4.5, 17.4, 3.8, 37.1, 26.9, 32.5,
    20.7, 9.2, 23.3, 74.2, 23.0, 33.3, 5.4, 29.4, 27.2, 6.4,
    38.6, 9.5, 20.0, 52.4, 23.0, 27.1, 19.2, 6.5, 17.9, 20.5,
    9.5, 15.4, 15.4, 21.0, 47.3, 20.0, 25.6, 44.8, 17.4, 22.3,
    26.9, 17.9, 12.8, 57.6, 12.8, 38.4, 19.2, 23.0, 33.3, 38.6,
    16.6, 13.1, 42.0, 5.1, 21.7, 5.1, 28.1, 26.9, 17.1, 24.1,
    22.0, 26.3, 11.7, 15.6, 5.1, 23.5, 33.3, 15.6, 27.4, 19.2,
    13.6, 14.6, 14.3, 14.1, 17.9, 7.7, 11.0, 38.6, 12.8, 19.2,
    28.4, 22.0, 13.6, 29.4, 52.4, 25.6, 29.4, 18.2, 29.4, 7.7
])

EDGE_DIAMS = np.array([
    11.4, 21.3, 19.7, 11.1, 10.6, 10.6, 3.7, 11.2, 8.8, 5.1,
    6.4, 7.0, 12.5, 10.0, 5.3, 7.5, 4.7,
    5.0, 5.9, 12.2,
    10.5, 7.4, 11.8, 6.6, 20.2, 6.1, 6.4, 4.7, 6.9, 8.1,
    4.9, 11.0, 6.9, 4.9, 4.6, 8.5, 4.6, 8.2, 5.5, 7.9,
    4.8, 10.1, 5.7, 5.7, 7.3, 7.8, 5.8, 7.5, 7.7, 7.2,
    7.5, 7.6, 9.0, 5.6,
    5.4, 8.1, 7.0, 7.4, 7.4, 6.4,
    4.2, 7.6, 7.3, 9.7, 8.1, 7.3, 7.0, 6.8, 7.3, 9.0,
    6.8, 6.5, 6.5, 7.4, 5.7, 8.9, 6.4, 7.5, 8.6, 8.5,
    8.6, 9.6, 6.2, 7.2, 8.8, 5.8, 7.3, 6.2, 7.1, 6.7,
    8.2, 9.7, 7.1, 5.8, 8.6, 10.9, 8.0, 6.5, 5.3, 6.9,
    5.8, 5.9, 6.0, 7.4, 6.3, 8.2, 8.4, 9.9, 11.1, 9.6,
    6.9, 5.1, 12.1, 6.9, 7.8, 7.7, 6.7, 8.3, 6.5, 5.7,
    5.2, 6.9, 6.8,
    5.8, 5.3, 6.5, 8.0, 10.2, 9.0, 6.3,
    8.9, 16.4, 8.1, 9.6, 10.8, 8.6, 12.6, 20.8, 5.9, 13.3,
    12.1, 11.6, 6.8, 6.4, 12.0, 7.4, 7.5, 12.2, 11.5, 8.1,
    9.7, 7.5, 9.4, 10.6, 12.5, 9.6, 8.0, 8.6, 7.9, 5.7,
    8.8, 13.4, 8.9, 7.6, 8.9, 9.9, 8.3, 11.6, 13.3, 7.7,
    6.9, 9.5, 4.5, 5.6, 4.3, 4.9, 3.4, 9.1, 9.4, 9.8,
    4.2,
    7.8, 9.0, 11.2,
    9.0, 8.1, 10.0, 8.3, 6.0, 4.7,
    10.5, 6.4, 11.7, 8.5,
    7.1, 5.9, 8.6, 2.0, 7.4, 9.0,
    9.8, 7.8, 9.0, 6.7, 8.7, 6.9, 14.7, 7.5, 7.7, 5.3,
    5.9, 10.7, 11.9, 7.4, 5.0, 7.4, 10.4, 11.5, 5.7, 12.2,
    7.4, 8.5, 9.0, 6.9, 11.4, 9.9, 13.5, 10.4, 10.6, 9.7,
    10.1, 7.2, 5.2, 7.7, 6.9, 7.7, 9.0, 7.1, 11.4, 5.0,
    9.6, 6.5, 7.6, 9.8, 12.7, 10.7, 4.8, 7.2, 12.3, 4.9,
    9.6, 6.1, 6.2, 6.6, 7.2, 8.1, 6.7, 6.9, 8.2, 10.0,
    4.7, 3.9, 12.1, 11.0, 5.7, 9.6, 8.4, 10.2, 7.5, 8.5,
    10.1, 10.4, 15.4, 8.6, 8.6, 7.2, 7.6, 8.2, 8.1, 7.8,
    9.4, 9.6, 7.7, 5.1, 5.1, 5.3, 7.7, 7.4, 5.3, 12.6,
    8.7, 13.4, 7.5, 9.0, 6.5, 7.0, 7.5, 7.5, 11.7, 9.0,
    6.5, 8.4, 7.3,
    9.5, 6.5, 6.6, 12.1, 11.0, 9.0, 5.8,
    11.1, 6.8, 6.5, 10.3, 9.1, 9.5, 5.6, 10.8, 10.5, 12.8
])

IN_NODE = 1
OUT_NODE = 195


def build_anatomy(DAA_m=None, DEA_m=None, L_AA=None, L_EA=None, p=None):
    """
    Build the full network arrays (src, trg, D, L) including the
    efferent arteriole exit segment.

    Parameters
    ----------
    DAA_m : float, optional
        Afferent arteriole diameter (um). Defaults to params["DAA_0"].
    DEA_m : float, optional
        Efferent arteriole diameter (um). Defaults to params["DEA_0"].
    L_AA : float, optional
        Afferent arteriole length (um). Defaults to params["L_AA"].
    L_EA : float, optional
        Efferent arteriole length (um). Defaults to params["L_EA"].
    p : dict, optional
        Parameter dict; uses global `params` if None.

    Returns
    -------
    src, trg : ndarray of int (1-indexed node IDs)
    D, L : ndarray of float (um)
    """
    if p is None:
        p = params
    if DAA_m is None:
        DAA_m = p["DAA_0"]
    if DEA_m is None:
        DEA_m = p["DEA_0"]
    if L_AA is None:
        L_AA = p["L_AA"]
    if L_EA is None:
        L_EA = p["L_EA"]

    # Add efferent exit edge (node 194 -> 195)
    src = np.append(SOURCE_NODES, 194)
    trg = np.append(TARGET_NODES, OUT_NODE)
    D = np.append(EDGE_DIAMS.copy(), DEA_m)
    L = np.append(EDGE_LENGTHS.copy(), L_EA)

    # Override afferent inlet edge (src==1) with DAA_m, L_AA
    in_mask = (src == IN_NODE)
    D[in_mask] = DAA_m
    L[in_mask] = L_AA

    return src, trg, D, L


# ============================================================
# Core network functions (ported from funcs_4.R)
# ============================================================

def _compute_R(mu_app, L, D, Rra, src, in_src):
    """
    Compute Hagen-Poiseuille resistance for each edge, adding
    pre-afferent resistance Rra to the inlet edge.

    R = 128*mu*L / (pi*D^4) * 1e3/133.32239/60   [nl/min/mmHg]
    """
    R = 128.0 * mu_app * L / (np.pi * D**4) * 1e3 / 133.32239 / 60.0
    in_mask = np.isin(src, in_src)
    R[in_mask] += Rra / 60.0
    return R


def get_P_filt(src, trg, L, D, mu_app, Rra, Rinf, Pa, Pe, P_bs,
               in_src, out_trg):
    """
    Solve for nodal pressures in the filtering capillary network.

    Uses the analytical solution for pressure distribution along a
    leaky tube (hyperbolic functions), then assembles and solves the
    linear system A*p = b for unknown interior node pressures.

    Returns
    -------
    p : ndarray
        Pressures at unknown (interior) nodes.
    u_nodes : ndarray
        The node IDs corresponding to entries of p.
    """
    # Unknown nodes: all nodes except in and out
    all_nodes = np.union1d(src, trg)
    u_nodes = np.sort(np.setdiff1d(all_nodes, np.array([in_src, out_trg])))
    lu = len(u_nodes)

    # Map node ID -> matrix row index (0-based)
    node_to_idx = {n: i for i, n in enumerate(u_nodes)}

    R = _compute_R(mu_app, L, D, Rra, src, [in_src])

    A = np.zeros((lu, lu))
    b = np.zeros(lu)
    bm = np.zeros(lu)

    for i, ni in enumerate(u_nodes):
        # Edges where ni is EITHER src or trg (boundary connections included)
        i_src_bd = np.where(src == ni)[0]  # edges where ni is source
        i_trg_bd = np.where(trg == ni)[0]  # edges where ni is target

        i_bd = np.concatenate([i_src_bd, i_trg_bd])  # all connected edges

        # Non-boundary connections (neither endpoint is in/out)
        i_src_nbd = np.where((src == ni) & (~np.isin(trg, [in_src, out_trg])))[0]
        i_trg_nbd = np.where((trg == ni) & (~np.isin(src, [in_src, out_trg])))[0]
        i_nbd = np.concatenate([i_src_nbd, i_trg_nbd])

        # Neighbouring node IDs (non-boundary)
        c_n = np.concatenate([trg[i_src_nbd], src[i_trg_nbd]])

        # a = sqrt(R / (L^2 * Rinf))
        a_bd = np.sqrt(R[i_bd] / (L[i_bd]**2 * Rinf[i_bd]))
        a_nbd = np.sqrt(R[i_nbd] / (L[i_nbd]**2 * Rinf[i_nbd]))

        aL_bd = a_bd * L[i_bd]
        aL_nbd = a_nbd * L[i_nbd]

        # Off-diagonal coefficient: dQ/dP_j for neighbour j
        coeff = a_nbd * L[i_nbd] / 2.0 / R[i_nbd] * (np.tanh(aL_nbd / 2)**2 - 1) / np.tanh(aL_nbd / 2)

        # Diagonal coefficient: sum of dQ/dP_i from all connected edges
        coeff_diag = a_bd * L[i_bd] / 2.0 / R[i_bd] * (np.tanh(aL_bd / 2)**2 + 1) / np.tanh(aL_bd / 2)

        # Bowman's space pressure contribution
        coeff_bs = np.tanh(aL_bd / 2) / (Rinf[i_bd] * aL_bd)

        # Aggregate off-diagonal by neighbour node
        if len(c_n) > 0:
            unique_cn = np.unique(c_n)
            for cn in unique_cn:
                mask = (c_n == cn)
                idx_cn = node_to_idx.get(cn)
                if idx_cn is not None:
                    A[i, idx_cn] += np.sum(coeff[mask])

        A[i, i] = np.sum(coeff_diag)
        b[i] = P_bs * np.sum(coeff_bs)

    # Update bm for boundary conditions (inlet Pa, outlet Pe)
    i_in_s = np.where(src == in_src)[0]
    i_in_t = np.where(trg == in_src)[0]
    i_out_s = np.where(src == out_trg)[0]
    i_out_t = np.where(trg == out_trg)[0]

    in_conns = np.concatenate([trg[i_in_s], src[i_in_t]])
    out_conns = np.concatenate([trg[i_out_s], src[i_out_t]])

    i_in = np.concatenate([i_in_s, i_in_t])
    i_out = np.concatenate([i_out_s, i_out_t])

    a_in = np.sqrt(R[i_in] / (L[i_in]**2 * Rinf[i_in]))
    a_out = np.sqrt(R[i_out] / (L[i_out]**2 * Rinf[i_out]))
    aL_in = a_in * L[i_in]
    aL_out = a_out * L[i_out]

    for k, cn in enumerate(in_conns):
        idx_cn = node_to_idx.get(cn)
        if idx_cn is not None:
            bm[idx_cn] -= Pa * a_in[k] * L[i_in[k]] / 2.0 / R[i_in[k]] * \
                (np.tanh(aL_in[k] / 2)**2 - 1) / np.tanh(aL_in[k] / 2)

    for k, cn in enumerate(out_conns):
        idx_cn = node_to_idx.get(cn)
        if idx_cn is not None:
            bm[idx_cn] -= Pe * a_out[k] * L[i_out[k]] / 2.0 / R[i_out[k]] * \
                (np.tanh(aL_out[k] / 2)**2 - 1) / np.tanh(aL_out[k] / 2)

    bb = b + bm
    p = solve(A, bb)

    return p, u_nodes


def run_net_one(src, trg, D, L, k, Ca_i, Paa_i, Pea_o, E, mu_app,
                Rra, Rinf, P_bs, t_wall, hpod, wpod, in_src, out_trg):
    """
    Run one pass of the network solver: given Rinf and mu_app, solve
    for pressures, flows, and filtration rates.

    Returns a dict with all per-edge quantities.
    """
    P_filt_p, u_nodes = get_P_filt(
        src=src, trg=trg, L=L, D=D, mu_app=mu_app,
        Rra=Rra, Rinf=Rinf, Pa=Paa_i, Pe=Pea_o,
        P_bs=P_bs, in_src=in_src, out_trg=out_trg
    )

    # Build pressure lookup: node_id -> pressure
    p_all_nodes = np.concatenate([[in_src], u_nodes, [out_trg]])
    p_all_pressures = np.concatenate([[Paa_i], P_filt_p, [Pea_o]])
    p_hash = dict(zip(p_all_nodes, p_all_pressures))

    p_src_arr = np.array([p_hash[n] for n in src])
    p_trg_arr = np.array([p_hash[n] for n in trg])

    R = _compute_R(mu_app, L, D, Rra, src, [in_src])
    a = np.sqrt(R / (L**2 * Rinf))

    # Correct flow direction: ensure p_src >= p_trg
    swap = p_src_arr < p_trg_arr
    src_c = src.copy()
    trg_c = trg.copy()
    src_c[swap] = trg[swap]
    trg_c[swap] = src[swap]
    p_src = np.array([p_hash[n] for n in src_c])
    p_trg = np.array([p_hash[n] for n in trg_c])

    # Mean pressure along each segment (Starling solution)
    p_avg = P_bs + np.tanh(a * L / 2) / (a * L / 2) * ((p_src + p_trg) / 2.0 - P_bs)

    # Total flow through each segment
    Q = (p_src - p_trg) / R

    # Flows at segment start (afferent, entering) and end (efferent, exiting)
    aL = a * L
    Q_src_a = -L / R * a / np.sinh(aL) * (-np.cosh(aL) * p_src + p_trg - P_bs * (1 - np.cosh(aL)))
    Q_trg_a = -L / R * a / np.sinh(aL) * (-p_src + np.cosh(aL) * p_trg + P_bs * (1 - np.cosh(aL)))

    # Filtration rate per segment (nl/min)
    Qfilt = (p_avg - P_bs) / Rinf

    # Protein concentration along the network (topological sort)
    Ca_arr, Ce_arr = get_C_filt(src_c, trg_c, Ca_i, Q_src_a - E, Q_trg_a - E, in_src, out_trg)

    # Wall thickness including podocyte
    t_new = t_wall + hpod / 2.0

    # Shear stress (dyne/cm^2) and hoop stress (kPa)
    shear = 32.0 * mu_app * Q / (np.pi * D**3) * 1e3 / 6.0
    hoop = (p_avg - P_bs) * D / (2.0 * t_new) * 133.32239

    return {
        "src": src_c, "trg": trg_c,
        "D": D, "L": L, "t": t_new,
        "mu": mu_app, "Rinf": Rinf, "k": k,
        "Pbs": np.full(len(src), P_bs),
        "Q": Q, "Qsm": 2*(p_src - p_avg)/R, "Qmt": 2*(p_avg - p_trg)/R,
        "Qsm_a": Q_src_a, "Qmt_a": Q_trg_a,
        "CSGFR": Qfilt,
        "Pa": p_src, "Pe": p_trg, "Pm": p_avg,
        "Ca": Ca_arr, "Ce": Ce_arr,
        "shear": shear, "hoop": hoop,
    }


def get_C_filt(src, trg, Ca_i, Qa, Qe, in_src, out_trg):
    """
    Compute plasma protein concentrations along the capillary network
    using a topological sort traversal.

    Ca: protein concentration at the SOURCE end of each edge
    Ce: protein concentration at the TARGET end of each edge
    """
    n_edges = len(src)
    Ca = np.full(n_edges, np.nan)
    Ce = np.full(n_edges, np.nan)

    # Identify inlet edges
    in_mask = np.isin(src, [in_src])
    Ca[in_mask] = Ca_i

    # Topological sort using BFS/DFS on the graph
    # Build adjacency: node -> list of outgoing edge indices
    node_order = _topo_sort(src, trg)

    for ni in node_order:
        i_trg = np.where(trg == ni)[0]  # edges where ni is target (incoming)
        i_src = np.where(src == ni)[0]  # edges where ni is source (outgoing)

        if len(i_trg) > 0 and len(i_src) > 0:
            Q_parents = Qe[i_trg]
            C_parents = Ce[i_trg]
            Ca_ni = np.sum(C_parents * Q_parents) / np.sum(Qa[i_src])
            Ca[i_src] = Ca_ni
            Ce[i_src] = Ca[i_src] * Qa[i_src] / Qe[i_src]
        elif len(i_trg) == 0 and len(i_src) > 0:
            # Inlet node: Ca already set
            Ce[i_src] = Ca[i_src] * Qa[i_src] / Qe[i_src]

    return Ca, Ce


def _topo_sort(src, trg):
    """
    Topological sort of nodes using Kahn's algorithm (BFS).
    Returns node IDs in topological order.
    """
    all_nodes = np.union1d(src, trg)
    # in-degree for each node
    in_deg = {n: 0 for n in all_nodes}
    children = {n: [] for n in all_nodes}
    for s, t in zip(src, trg):
        in_deg[t] += 1
        children[s].append(t)

    queue = [n for n in all_nodes if in_deg[n] == 0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for ch in children[n]:
            in_deg[ch] -= 1
            if in_deg[ch] == 0:
                queue.append(ch)
    return order


def hemato_helper(E, QB, D):
    """
    Compute tube hematocrit (H_t) and discharge hematocrit (H_d)
    from erythrocyte volume flow E and blood flow QB.
    """
    H_t = E / (QB + 1e-30)
    H_d = H_t / (0.5 * (1 + np.exp(-0.633 * (D / 10.43 - 1))))
    return H_t, H_d


def get_E_filt(src, trg, D, Ea, Qa, Qe, in_src, out_trg):
    """
    Compute erythrocyte flow (E) through each network segment via
    topological sort, accounting for bifurcation partitioning.

    Returns E, H_t, H_d arrays (per edge).
    """
    n_edges = len(src)
    E = np.full(n_edges, np.nan)

    in_mask = np.isin(src, [in_src])
    E[in_mask] = Ea

    H_t, H_d = hemato_helper(E, (Qa + Qe) / 2.0, D)

    node_order = _topo_sort(src, trg)

    for ni in node_order:
        i_trg = np.where(trg == ni)[0]  # incoming edges
        i_src = np.where(src == ni)[0]  # outgoing edges

        if len(i_trg) == 0:
            Esum = Ea
        else:
            Esum = np.nansum(E[i_trg])

        if len(i_trg) > 0:
            Q_e_parents = Qe[i_trg]
            H_d_parents = H_d[i_trg]
            denom = np.sum(Q_e_parents)
            H_d_parent = np.sum(H_d_parents * Q_e_parents) / denom if denom > 0 else 0
            D_parent = np.sqrt(np.sum(D[i_trg]**2))
        else:
            H_d_parent = H_d[in_mask][0] if np.any(in_mask) else 0
            D_parent = D[in_mask][0] if np.any(in_mask) else 1

        if len(i_src) == 0:
            pass
        elif len(i_src) == 1:
            E[i_src[0]] = Esum
        elif len(i_src) == 2:
            X0 = 0.4 / D_parent
            FQ = Qa[i_src] / np.sum(Qa[i_src])
            xt = (FQ - X0) / (1 - 2 * X0)
            xt = np.where(xt < 0, 0.0, xt)
            A_vec = np.array([
                -6.96 * np.log(D[i_src[0]] / D[i_src[1]]) / D_parent,
                -6.96 * np.log(D[i_src[1]] / D[i_src[0]]) / D_parent
            ])
            B = 1 + 6.98 * (1 - H_d_parent) / D_parent
            # Avoid log(0)
            xt_safe = np.where(xt <= 0, 1e-10, np.where(xt >= 1, 1 - 1e-10, xt))
            E[i_src] = Esum / (1 + np.exp(A_vec - B * np.log(xt_safe / (1 - xt_safe))))
        else:
            # More than 2 branches: sequential pairwise partition
            i_s_t = list(i_src)
            li = len(i_s_t) - 1
            for _ in range(li):
                D1 = D[i_s_t[0]]
                D2 = np.sqrt(np.sum(D[i_s_t[1:]]**2))
                FQ = np.array([
                    Qa[i_s_t[0]] / np.sum(Qa[i_s_t]),
                    np.sum(Qa[i_s_t[1:]]) / np.sum(Qa[i_s_t])
                ])
                X0 = 0.4 / D_parent
                xt = (FQ - X0) / (1 - 2 * X0)
                xt = np.where(xt < 0, 0.0, xt)
                A_vec = np.array([
                    -6.96 * np.log(D1 / D2) / D_parent,
                    -6.96 * np.log(D2 / D1) / D_parent
                ])
                B = 1 + 6.98 * (1 - H_d_parent) / D_parent
                xt_safe = np.where(xt <= 0, 1e-10, np.where(xt >= 1, 1 - 1e-10, xt))
                E_temp = Esum / (1 + np.exp(A_vec - B * np.log(xt_safe / (1 - xt_safe))))
                E[i_s_t[0]] = E_temp[0]
                Esum = E_temp[1]
                i_s_t = i_s_t[1:]
            E[i_s_t[0]] = Esum

        H_t, H_d = hemato_helper(E, (Qa + Qe) / 2.0, D)

    return E, H_t, H_d


def get_mu_app(H_d, D, mu_plas):
    """
    Compute apparent viscosity using the Pries empirical formula.

    H_d : discharge hematocrit per edge
    D   : diameter (um) per edge
    mu_plas : plasma viscosity (cP) per edge
    """
    H_d = np.where(H_d > 1, 0.99, H_d)
    eta045 = 220 * np.exp(-1.3 * D) + 3.2 - 2.44 * np.exp(-0.06 * D**0.645)
    gamma = (0.8 + np.exp(-0.075 * D)) * (-1 + 1 / (1 + D**12 / 1e11)) + 1 / (1 + D**12 / 1e11)
    xi = 1 + (eta045 - 1) * ((1 - H_d)**gamma - 1) / (0.55**gamma - 1)
    return xi  # relative to water; multiplied by mu_plas inside this formula returns cP


def get_Rinf(src, trg, k, D, L, Ca, Ce, Pa_e, Pe_e, Pbs, Qa, Rinf_prev,
             E, mu, Rra, in_src, out_trg, num_quad=1000):
    """
    Compute updated filtration resistance Rinf for each segment using
    Starling's law with oncotic pressure feedback.

    Returns dict with Rinf, Pi_int, p_int, Ce (exit concentrations).
    """
    R = _compute_R(mu, L, D, Rra, src, [in_src])
    a = np.sqrt(R / (L**2 * Rinf_prev))

    # Mean transmural pressure (used for filtration resistance)
    p_diff_int = np.tanh(a * L / 2) / (a * L / 2) * ((Pa_e + Pe_e) / 2.0 - Pbs)

    # Concentration profile along each segment via quadrature
    x_mat = np.arange(0, num_quad + 1, dtype=float)  # shape (num_quad+1,)

    # Per-segment a*L/num_quad
    aL_n = (a * L / num_quad)[:, np.newaxis]  # (n_edges, 1)
    ax = aL_n * x_mat[np.newaxis, :]           # (n_edges, num_quad+1)

    Pi_term = np.sinh(ax) - (np.cosh(a * L) / np.sinh(a * L))[:, np.newaxis] * np.cosh(ax)
    Pj_term = np.cosh(ax) / np.sinh(a * L)[:, np.newaxis]
    Pbs_term = np.sinh(ax) + ((1 - np.cosh(a * L)) / np.sinh(a * L))[:, np.newaxis] * np.cosh(ax)

    dpdx_x = (a * Pa_e)[:, np.newaxis] * Pi_term + \
              (a * Pe_e)[:, np.newaxis] * Pj_term - \
              (a * Pbs)[:, np.newaxis] * Pbs_term

    Q_x = -(L / R)[:, np.newaxis] * dpdx_x

    Q0 = -L / R * a / np.sinh(a * L) * (-np.cosh(a * L) * Pa_e + Pe_e - (1 - np.cosh(a * L)) * Pbs)

    Q_ratio = (Q0 - E)[:, np.newaxis] / (Q_x - E[:, np.newaxis])
    C_x = Ca[:, np.newaxis] * Q_ratio

    # Oncotic pressure (van't Hoff / Landis formula)
    a1, a2, a3 = 2.1, 0.16, 0.009
    Pi_x = a1 * C_x + a2 * C_x**2 + a3 * C_x**3

    # Trapezoidal quadrature for mean oncotic pressure
    Pi_int = 1.0 / (num_quad + 1) * (
        0.5 * (Pi_x[:, 0] + Pi_x[:, num_quad]) + np.sum(Pi_x[:, 1:num_quad], axis=1)
    )

    Rinf_new = 1.0 / (k * L * np.pi * D * (1 - Pi_int / p_diff_int))

    Ce_out = C_x[:, num_quad]

    return {
        "Rinf": Rinf_new,
        "Pi_int": Pi_int,
        "p_int": p_diff_int,
        "Ce": Ce_out,
    }


def run_glom(src, trg, D, L, k, Rinf_init, Paa_i, Pea_o, P_bs,
             Ca_i, H_t_sys, mu_plas, Rra, t_wall, hpod, wpod,
             in_src, out_trg,
             num_iter=150, mu_tol=1e-3, Rinf_tol=1e-3, beta=3,
             verbose=False):
    """
    Iterative solver for the glomerular capillary network.

    Alternates between:
    1. Solving the pressure/flow network with current Rinf and mu_app
    2. Updating Rinf using get_Rinf (Starling + oncotic feedback)
    3. Updating mu_app using get_mu_app (Pries formula)

    Returns
    -------
    dict with keys: G (per-edge results), SNGFR, FF, num_iter, ...
    """
    count = 0
    SNGFR0 = 0.0
    Rinf0 = Rinf_init.copy()
    mu0 = mu_plas * np.ones(len(src))

    mu_err = mu_tol * 5
    Rinf_err = Rinf_tol * 5

    # Initial network pass (E=0, plasma viscosity)
    G = run_net_one(
        src=src, trg=trg, D=D, L=L, k=k, Ca_i=Ca_i,
        Paa_i=Paa_i, Pea_o=Pea_o, E=0.0,
        mu_app=mu0, Rra=Rra, Rinf=Rinf0,
        P_bs=P_bs, t_wall=t_wall, hpod=hpod, wpod=wpod,
        in_src=in_src, out_trg=out_trg
    )

    while (mu_err > mu_tol or Rinf_err > Rinf_tol) and count < num_iter:
        # Erythrocyte flow distribution
        in_mask = np.isin(src, [in_src])
        Ea = H_t_sys * G["Q"][in_mask][0]
        E_filt, H_t, H_d = get_E_filt(
            src=G["src"], trg=G["trg"], D=D, Ea=Ea,
            Qa=G["Qsm_a"], Qe=G["Qmt_a"],
            in_src=in_src, out_trg=out_trg
        )

        # Plasma viscosity (concentration-weighted)
        mu_plas_func = 0.274 + 0.177 * (G["Ca"] * G["Qsm_a"] / (G["Q"] + 1e-30))
        mu_plas_func = np.where(mu_plas_func < 0, mu_plas, mu_plas_func)

        # Apparent viscosity
        mu_app_inf = get_mu_app(H_d=H_d, D=D, mu_plas=mu_plas_func)

        # Updated Rinf
        Rinf_res = get_Rinf(
            src=G["src"], trg=G["trg"], k=k, D=D, L=L,
            Ca=G["Ca"], Ce=G["Ce"],
            Pa_e=G["Pa"], Pe_e=G["Pe"], Pbs=G["Pbs"],
            Qa=G["Qsm_a"], Rinf_prev=G["Rinf"],
            E=E_filt, mu=mu0, Rra=Rra,
            in_src=in_src, out_trg=out_trg, num_quad=1000
        )

        Rinf1 = Rinf_res["Rinf"]
        Rinf_prev = G["Rinf"].copy()

        # Flag problematic segments
        i_b = np.where((Rinf1 < 0) | (Rinf_res["Ce"] < 0) & (Rinf_prev < 1e10))[0]
        i_c = np.where((Rinf1 < 0) | (Rinf_res["Ce"] < 0) & (Rinf_prev >= 1e10))[0]

        # Relaxation factor
        valid = Rinf1 > 0
        if np.any(valid):
            alpha_R = np.max(np.abs(Rinf1[valid] - Rinf_prev[valid]) / Rinf_prev[valid]) * beta
        else:
            alpha_R = 1.0
        alpha_mu = np.max(np.abs(mu_app_inf - mu0) / mu0) * beta
        alpha = max(alpha_R, alpha_mu, 1.0)

        mu_app = mu0 + (mu_app_inf - mu0) / alpha
        Rinf_new = Rinf_prev + (Rinf1 - Rinf_prev) / alpha
        Rinf_new[i_b] = (1 + 1.0 / beta) * Rinf_prev[i_b]
        if len(i_c) > 0:
            Rinf_new[i_c] = Rinf_prev[i_c]

        # Inlet/outlet edges are non-filtering (very high Rinf)
        io_mask = np.isin(src, [in_src]) | np.isin(trg, [out_trg])
        Rinf_new[io_mask] = 1e20

        G = run_net_one(
            src=src, trg=trg, D=D, L=L, k=k, Ca_i=Ca_i,
            Paa_i=Paa_i, Pea_o=Pea_o, E=E_filt,
            mu_app=mu_app, Rra=Rra, Rinf=Rinf_new,
            P_bs=P_bs, t_wall=t_wall, hpod=hpod, wpod=wpod,
            in_src=in_src, out_trg=out_trg
        )

        SNGFR = np.sum(G["CSGFR"])
        SNGFR_err = abs(SNGFR - SNGFR0) / (SNGFR0 + 1e-30)
        Rinf_err = np.max(np.abs(Rinf_new - Rinf0) / Rinf0)
        mu_err = np.max(np.abs(mu_app - mu0) / mu0)

        SNGFR0 = SNGFR
        Rinf0 = Rinf_new.copy()
        mu0 = mu_app.copy()
        count += 1

        if verbose and count % 25 == 0:
            print(f"  Iteration {count}: SNGFR={SNGFR:.4f}, Rinf_err={Rinf_err:.2e}, mu_err={mu_err:.2e}")

    SNGFR = np.sum(G["CSGFR"])

    # Final erythrocyte pass for FF
    in_mask = np.isin(src, [in_src])
    Ea = H_t_sys * G["Q"][in_mask][0]
    E_filt1, _, _ = get_E_filt(
        src=G["src"], trg=G["trg"], D=D, Ea=Ea,
        Qa=G["Qsm_a"], Qe=G["Qmt_a"],
        in_src=in_src, out_trg=out_trg
    )

    Q_plasma = G["Q"][in_mask][0] - E_filt1[in_mask][0]
    FF = SNGFR / Q_plasma if Q_plasma > 0 else np.nan

    if verbose:
        if count >= num_iter:
            print(f"  Max iterations reached ({num_iter}). SNGFR_err={SNGFR_err:.2e}")
        else:
            print(f"  Converged in {count} iterations. SNGFR={SNGFR:.4f} nl/min")

    return {
        "G": G,
        "SNGFR": SNGFR,
        "FF": FF,
        "num_iter": count,
        "Rinf_tol": Rinf_tol,
    }


def run_baseline(Pa=100.0, p=None, verbose=False):
    """
    Run the glomerular capillary network at baseline conditions.

    Parameters
    ----------
    Pa : float
        Afferent arteriole inlet pressure (mmHg). Default 100.
    p : dict, optional
        Parameter dict; uses global `params` if None.
    verbose : bool
        Print iteration progress.

    Returns
    -------
    dict with keys:
        SNGFR      : float, nl/min
        Pgc_mean   : float, mmHg  (mean capillary pressure, excluding inlet/outlet)
        Pgc_inlet  : float, mmHg  (pressure at efferent side of afferent arteriole exit)
        Q          : float, nl/min (total plasma flow)
        FF         : float         (filtration fraction)
        G          : dict          (per-edge results)
        num_iter   : int
    """
    if p is None:
        p = params

    src, trg, D, L = build_anatomy(p=p)

    # Initial Rinf (from R: 1/(k*D*L*pi*0.1))
    # NOTE: io edges NOT set to 1e20 here — run_glom sets them to 1e20
    # inside the loop after the first iteration, matching R's run_network.R
    k_arr = p["k"] * np.ones(len(src))
    Rinf0 = 1.0 / (k_arr * D * L * np.pi * 0.1)

    result = run_glom(
        src=src, trg=trg, D=D, L=L,
        k=k_arr,
        Rinf_init=Rinf0,
        Paa_i=Pa,
        Pea_o=p["Pea_o"],
        P_bs=p["P_bs"],
        Ca_i=p["Ca_i"],
        H_t_sys=p["H_t_sys"],
        mu_plas=p["mu_plas"],
        Rra=p["Rra"],
        t_wall=p["t"],
        hpod=p["hpod"],
        wpod=p["wpod"],
        in_src=IN_NODE,
        out_trg=OUT_NODE,
        num_iter=p["num_iter"],
        mu_tol=p["mu_tol"],
        Rinf_tol=p["Rinf_tol"],
        beta=p["beta"],
        verbose=verbose,
    )

    G = result["G"]

    # Pgc_mean: mean capillary pressure excluding inlet (src=1) and outlet (trg=195) edges
    in_mask = np.isin(src, [IN_NODE])
    out_mask = np.isin(trg, [OUT_NODE])
    cap_mask = ~in_mask & ~out_mask
    Pgc_mean = np.mean(G["Pm"][cap_mask])

    # Pgc_inlet: pressure at the exit of the afferent arteriole = Pe of the inlet edge
    Pgc_inlet = G["Pe"][in_mask][0]

    # Total plasma flow (from inlet edge)
    Q_inlet = G["Q"][in_mask][0]

    return {
        "SNGFR": result["SNGFR"],
        "Pgc_mean": Pgc_mean,
        "Pgc_inlet": Pgc_inlet,
        "Q": Q_inlet,
        "FF": result["FF"],
        "G": G,
        "num_iter": result["num_iter"],
    }


# ============================================================
# Surrogate Glomerulus Lookup (glom_SS equivalent)
# ============================================================

_SURR_DF = None  # lazy-loaded cache
_SURR_PATH = None


def load_surr_glom_df(rds_path=None):
    """
    Load the precomputed surrogate glomerulus lookup table.

    Parameters
    ----------
    rds_path : str, optional
        Path to the surr_glom_df_20220724.RDS file.
        If None, searches relative to this file's location.

    Returns
    -------
    pandas.DataFrame with columns: P.in, D.in, Ca.in, Pavg, SNGFR, Q
    """
    global _SURR_DF, _SURR_PATH

    if rds_path is None:
        import os
        this_dir = os.path.dirname(os.path.abspath(__file__))
        # data/ folder sits alongside this model.py
        candidate = os.path.join(
            this_dir, "data",
            "surr_glom_df_20220724.RDS"
        )
        rds_path = os.path.normpath(candidate)

    if _SURR_DF is not None and _SURR_PATH == rds_path:
        return _SURR_DF

    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "pyreadr is required to read RDS files. "
            "Install with: pip install pyreadr"
        )

    result = pyreadr.read_r(rds_path)
    df = result[None]
    _SURR_DF = df
    _SURR_PATH = rds_path
    return df


def glom_SS(Pa, Ca, D, surr_glom_df=None, rds_path=None):
    """
    Surrogate glomerulus steady-state lookup (R glom_SS equivalent).

    Looks up Q, SNGFR, and Pavg from the precomputed table for given
    (Pa, D, Ca) conditions, with linear interpolation.

    Parameters
    ----------
    Pa : float
        Afferent arteriole inlet pressure (mmHg).
    Ca : float
        Plasma protein concentration (g/dl).
    D  : float
        Afferent arteriole diameter (um).
    surr_glom_df : pd.DataFrame, optional
        Preloaded surrogate table. If None, loads from rds_path.
    rds_path : str, optional
        Path to RDS file (used if surr_glom_df is None).

    Returns
    -------
    dict with keys: Q (nl/min), SNGFR (nl/min), Pavg (mmHg)
    """
    if surr_glom_df is None:
        surr_glom_df = load_surr_glom_df(rds_path)

    df = surr_glom_df

    # Filter exactly as R glom_SS does (strict inequalities):
    #   indx1 = P.in < (Pa + 2.5) & P.in > (Pa - 2.5)
    #   indx2 = round(D.in - D, 2) < 0.05 & round(D.in - D, 2) > -0.05
    mask_P = (df['P.in'] < (Pa + 2.5)) & (df['P.in'] > (Pa - 2.5))
    d_diff_rounded = (df['D.in'] - D).round(2)
    mask_D = (d_diff_rounded < 0.05) & (d_diff_rounded > -0.05)
    indx = df[mask_P & mask_D]

    if len(indx) == 0:
        raise ValueError(f"No entry found in surrogate table for Pa={Pa}, D={D}, Ca={Ca}")

    if len(indx) == 1:
        row = indx.iloc[0]
        return {"Q": row["Q"], "SNGFR": row["SNGFR"], "Pavg": row["Pavg"]}

    if len(indx) == 2:
        # Linear interpolation along the single varying dimension
        input_vec = [Pa, D, Ca]
        u = [
            round(abs(indx.iloc[0, 0] - indx.iloc[1, 0]), 2),
            round(abs(indx.iloc[0, 1] - indx.iloc[1, 1]), 2),
            round(abs(indx.iloc[0, 2] - indx.iloc[1, 2]), 2),
        ]
        varying = [i for i, v in enumerate(u) if v != 0]
        if not varying:
            row = indx.iloc[0]
            return {"Q": row["Q"], "SNGFR": row["SNGFR"], "Pavg": row["Pavg"]}
        j = varying[0]
        xi = input_vec[j]
        col_name = ["P.in", "D.in", "Ca.in"][j]
        x_arr = indx[col_name].values
        # Linear interpolation
        t = (xi - x_arr[0]) / (x_arr[1] - x_arr[0])
        Q = indx["Q"].values[0] + t * (indx["Q"].values[1] - indx["Q"].values[0])
        SNGFR = indx["SNGFR"].values[0] + t * (indx["SNGFR"].values[1] - indx["SNGFR"].values[0])
        Pavg = indx["Pavg"].values[0] + t * (indx["Pavg"].values[1] - indx["Pavg"].values[0])
        return {"Q": Q, "SNGFR": SNGFR, "Pavg": Pavg}

    if len(indx) == 4:
        # Bilinear interpolation over 2 varying dimensions
        input_vec = [Pa, D, Ca]
        l_length = [indx.iloc[:, j].nunique() for j in range(3)]
        varying = [j for j, v in enumerate(l_length) if v > 1]
        xi = [input_vec[j] for j in varying]
        col_names = ["P.in", "D.in", "Ca.in"]
        xs = sorted(indx[col_names[varying[0]]].unique())
        ys = sorted(indx[col_names[varying[1]]].unique())
        # Grid interpolation
        tx = (xi[0] - xs[0]) / (xs[1] - xs[0])
        ty = (xi[1] - ys[0]) / (ys[1] - ys[0])
        # Get 4 corners
        def _get(px, py, col):
            row = indx[(abs(indx[col_names[varying[0]]] - px) < 1e-10) &
                       (abs(indx[col_names[varying[1]]] - py) < 1e-10)]
            return row[col].values[0]
        Q = ((1-tx)*(1-ty)*_get(xs[0],ys[0],"Q") + tx*(1-ty)*_get(xs[1],ys[0],"Q") +
             (1-tx)*ty*_get(xs[0],ys[1],"Q") + tx*ty*_get(xs[1],ys[1],"Q"))
        SNGFR = ((1-tx)*(1-ty)*_get(xs[0],ys[0],"SNGFR") + tx*(1-ty)*_get(xs[1],ys[0],"SNGFR") +
                 (1-tx)*ty*_get(xs[0],ys[1],"SNGFR") + tx*ty*_get(xs[1],ys[1],"SNGFR"))
        Pavg = ((1-tx)*(1-ty)*_get(xs[0],ys[0],"Pavg") + tx*(1-ty)*_get(xs[1],ys[0],"Pavg") +
                (1-tx)*ty*_get(xs[0],ys[1],"Pavg") + tx*ty*_get(xs[1],ys[1],"Pavg"))
        return {"Q": Q, "SNGFR": SNGFR, "Pavg": Pavg}

    # Fall through: nearest neighbour if none of the above
    row = indx.iloc[0]
    return {"Q": row["Q"], "SNGFR": row["SNGFR"], "Pavg": row["Pavg"]}


def run_baseline_surrogate(Pa=100.0, D=None, Ca=None, p=None, rds_path=None):
    """
    Run the SURROGATE LOOKUP baseline (fast, exact match to R run_baseline.R).

    Uses the precomputed glom_SS table from surr_glom_df_20220724.RDS.

    Parameters
    ----------
    Pa : float
        Afferent arteriole inlet pressure (mmHg). Default 100.
    D : float, optional
        Arteriole diameter (um). Defaults to params["DAA_0"] = 7.
    Ca : float, optional
        Plasma protein (g/dl). Defaults to params["Ca_i"].
    p : dict, optional
        Parameter dict; uses global `params` if None.
    rds_path : str, optional
        Path to RDS surrogate file.

    Returns
    -------
    dict with keys:
        SNGFR      : float, nl/min
        Pgc_inlet  : float, mmHg  (= 2*Pavg - Pa, from surrogate)
        Q          : float, nl/min (total glomerular flow)
        Pavg       : float, mmHg  (mean pressure from surrogate)
        FF         : float         (filtration fraction SNGFR/Q)
    """
    if p is None:
        p = params
    if D is None:
        D = p["DAA_0"]
    if Ca is None:
        Ca = p["Ca_i"]

    GG = glom_SS(Pa=Pa, Ca=Ca, D=D, rds_path=rds_path)

    SNGFR = GG["SNGFR"]
    Q = GG["Q"]
    Pavg = GG["Pavg"]

    # Pgc_inlet: estimated from Pavg (see run_baseline.R line 38)
    # Pavg from glom_SS is the mean of p_src+p_trg for the inlet edge in the network
    # Pgc_inlet = 2*Pavg - Pa  (from pressure symmetry at the afferent exit)
    Pgc_inlet = 2.0 * Pavg - Pa

    FF = SNGFR / Q if Q > 0 else np.nan

    return {
        "SNGFR": SNGFR,
        "Pgc_inlet": Pgc_inlet,
        "Q": Q,
        "Pavg": Pavg,
        "FF": FF,
    }


if __name__ == "__main__":
    print("M002 Richfield 2024 — Surrogate lookup baseline (Pa=100 mmHg)")
    print("=" * 60)
    res = run_baseline_surrogate(Pa=100.0)
    print(f"\nSNGFR     = {res['SNGFR']:.4f} nl/min  (ref: 29.71)")
    print(f"Pgc_inlet = {res['Pgc_inlet']:.4f} mmHg    (ref: 51.18)")
    print(f"Q         = {res['Q']:.4f} nl/min  (ref: 101.61)")
    print(f"FF        = {res['FF']:.4f}          (ref: 0.292)")
    print()
    print("Full network solver baseline (Pa=100 mmHg):")
    res2 = run_baseline(Pa=100.0, verbose=False)
    print(f"SNGFR     = {res2['SNGFR']:.4f} nl/min  (ref: 29.71)")
    print(f"Pgc_mean  = {res2['Pgc_mean']:.4f} mmHg    (ref: 49.88)")
    print(f"Pgc_inlet = {res2['Pgc_inlet']:.4f} mmHg    (ref: 51.18)")
    print(f"FF        = {res2['FF']:.4f}          (ref: 0.292)")
