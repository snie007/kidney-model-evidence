#!/usr/bin/env python3
"""
replicate_fig5.py (evidence-repo version) — M006 Maass 2019 Figure 5.

Uses data/ and evidence/ paths relative to script directory.
See REPLICATION_LOG.md for full details.
"""
import os, sys, json, datetime, subprocess
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(_script_dir, 'data')
EVIDENCE_DIR = os.path.join(_script_dir, 'evidence')
sys.path.insert(0, _script_dir)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import simulate_cisplatin, get_outputs, THETA, IDX
from model import _invitro_shed_rate, _mps_shed_to_human_rate, _neutrophil_kim1_factor


def _git_hash():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_script_dir,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return 'unknown'


def run_population(n_virt=100, immune=True, seed=42, t_end_h=240.0, n_points=501):
    from scipy.interpolate import interp1d
    rng = np.random.default_rng(seed)
    t_common = np.linspace(0.0, t_end_h, n_points)
    all_C_KIM1_plasma = np.zeros((n_virt, n_points))
    all_C_KIM1_urine  = np.zeros((n_virt, n_points))
    all_C_drug_plasma = np.zeros((n_virt, n_points))

    for i in range(n_virt):
        theta_i = dict(THETA)
        BW_i = max(40.0, rng.normal(70.0, 12.0))
        theta_i['BW']   = BW_i
        theta_i['BSA']  = 0.007184 * (BW_i ** 0.425) * (175.0 ** 0.725)
        theta_i['CO']   = THETA['CO'] * (BW_i / 70.0) ** 0.75
        vol_sc = BW_i / 70.0
        for vk in ['V_plasma', 'V_kidney', 'V_liver', 'V_muscle', 'V_rest']:
            theta_i[vk] = THETA[vk] * vol_sc
        theta_i['V_dist_plasma'] = theta_i['V_plasma']
        theta_i['N_nephrons']    = max(5e5, rng.normal(1.8e6, 4.5e5))
        theta_i['C_KIM1_plasma_baseline'] = max(5.0, rng.normal(50.0, 25.0))

        try:
            t_i, y_i, _ = simulate_cisplatin(theta=theta_i, immune=immune,
                                              t_end_h=t_end_h, n_points=n_points)
            def safe_interp(ts, ys):
                f = interp1d(ts, ys, bounds_error=False,
                             fill_value=(ys[0], ys[-1]))
                return f(t_common)

            C_KIM1_p = y_i[IDX['A_KIM1_p']] / (theta_i['V_dist_plasma'] * 1000.0)
            Q_u = theta_i['Q_urine_Lph']
            f_u = theta_i['f_urine']
            V_k = theta_i['V_kidney']
            C_KIM1_u = np.zeros(len(t_i))
            for j, tj in enumerate(t_i):
                C_k_j = max(y_i[IDX['A_kidney'], j], 0.0) / V_k
                C_inv = _invitro_shed_rate(C_k_j, max(0.0, tj), theta_i)
                R_tot = _mps_shed_to_human_rate(C_inv, theta_i)
                neut  = _neutrophil_kim1_factor(tj, theta_i) if (immune and tj > 0) else 1.0
                C_KIM1_u[j] = f_u * R_tot * neut / (Q_u * 1000.0)

            all_C_KIM1_plasma[i] = safe_interp(t_i, C_KIM1_p)
            all_C_KIM1_urine[i]  = safe_interp(t_i, C_KIM1_u)
            all_C_drug_plasma[i] = safe_interp(t_i,
                y_i[IDX['A_plasma']] / theta_i['V_dist_plasma'])
        except Exception as e:
            print(f"  VP {i} failed: {e}")
            all_C_KIM1_plasma[i, :] = np.nan
            all_C_KIM1_urine[i, :]  = np.nan

    return t_common, all_C_KIM1_plasma, all_C_KIM1_urine, all_C_drug_plasma


def pop_stats(arr):
    mn = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    return mn, mn-sd, mn+sd, mn-2*sd, mn+2*sd


def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"M006 Figure 5 replication (evidence repo) — {ts}")

    t_no, p_no, u_no, pd_no = run_population(n_virt=100, immune=False)
    t_im, p_im, u_im, pd_im = run_population(n_virt=100, immune=True)

    mn_p_no, lo1_pno, hi1_pno, lo2_pno, hi2_pno = pop_stats(p_no)
    mn_p_im, lo1_pim, hi1_pim, lo2_pim, hi2_pim = pop_stats(p_im)
    mn_u_im, lo1_uim, hi1_uim, lo2_uim, hi2_uim = pop_stats(u_im)

    # Pass/fail
    baseline = THETA['C_KIM1_plasma_baseline']
    peak_no  = float(np.max(mn_p_no))
    peak_im  = float(np.max(mn_p_im))
    peak_u   = float(np.max(mn_u_im))
    t_peak   = float(t_im[np.argmax(mn_p_im)])

    pf = {
        'plasma_no_immune_peak_pgml': round(peak_no, 1),
        'crit1_2x_baseline': bool(peak_no >= 2.0 * baseline),
        'plasma_immune_peak_pgml': round(peak_im, 1),
        'crit2_plasma_immune_range': bool(300 <= peak_im <= 3000),
        'urine_immune_peak_pgml': round(peak_u, 1),
        'crit3_urine_immune_range': bool(1000 <= peak_u <= 10000),
        'plasma_peak_time_h': round(t_peak, 1),
        'crit4_timing': bool(8 <= t_peak <= 48),
    }
    n_pass = sum([pf['crit1_2x_baseline'], pf['crit2_plasma_immune_range'],
                  pf['crit3_urine_immune_range'], pf['crit4_timing']])
    status = 'INFORMATIVE_PASS' if n_pass == 4 else ('INFORMATIVE_FAIL' if n_pass >= 2 else 'FAIL')
    pf['status'] = status
    pf['n_pass'] = n_pass

    print(f"  Status: {status} ({n_pass}/4)")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('M006 — Maass 2019 Figure 5 (Cisplatin, N=100 virtual patients)',
                 fontsize=10, fontweight='bold')

    for ax, mn, lo1, hi1, lo2, hi2, t_vec, title, ymax, col in [
        (axes[0], mn_p_no, lo1_pno, hi1_pno, lo2_pno, hi2_pno, t_no,
         '(a) Plasma KIM-1\n(no immune effect)', max(300.0, peak_no*1.5), '#888888'),
        (axes[1], mn_p_im, lo1_pim, hi1_pim, lo2_pim, hi2_pim, t_im,
         '(b) Plasma KIM-1\n(with neutrophil recruitment)', max(2000.0, peak_im*1.5), '#886644'),
        (axes[2], mn_u_im, lo1_uim, hi1_uim, lo2_uim, hi2_uim, t_im,
         '(c) Urine KIM-1\n(with neutrophil recruitment)', max(6000.0, peak_u*1.5), '#4488aa'),
    ]:
        ax.fill_between(t_vec, lo2, hi2, color=col, alpha=0.2)
        ax.fill_between(t_vec, lo1, hi1, color=col, alpha=0.4)
        ax.plot(t_vec, mn, 'r-', lw=2, label=f'Pop. mean (N=100)\nPeak: {mn.max():.0f} pg/mL')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('KIM-1 (pg/mL)')
        ax.set_title(title, fontsize=9)
        ax.set_xlim([0, t_vec[-1]])
        ax.set_ylim([0, ymax])
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    axes[1].axhline(1000, color='k', ls='--', lw=1, alpha=0.5, label='~1000 pg/mL (paper)')
    axes[2].axhline(3000, color='k', ls='--', lw=1, alpha=0.5, label='~3000 pg/mL (paper)')

    fig_path = os.path.join(EVIDENCE_DIR, 'M006_fig5.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {fig_path}")

    # Artifact
    art = dict(
        model_id='M006', pmid='30869201',
        timestamp=ts, git_commit=_git_hash(),
        input_csvs=[os.path.join(DATA_DIR, 'M006_PMID30869201_fig3a_cisplatin.csv')],
        pass_fail=pf,
        notes='Evidence-repo version. Supplementary equations S2 not accessible.'
    )
    art_path = os.path.join(EVIDENCE_DIR, f'M006_fig5_{ts}.json')
    with open(art_path, 'w') as f:
        json.dump(art, f, indent=2)
    print(f"  Artifact: {art_path}")

    return 0 if 'PASS' in status else 1


if __name__ == '__main__':
    sys.exit(main())
