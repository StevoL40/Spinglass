"""Generate the README's SK phase diagram with RS and RSB boundaries.

Run from the project root: python python/plot_sk_phase_diagram.py
Uses independent adaptive Gaussian integration, not the dynamical solver.
The vertical SG/mixed boundary is the known full-RSB result, not a numerical
solution of the Parisi equations. The dashed RS continuation is not physical.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def moments(temperature, field, q):
    def average(kind):
        def integrand(z):
            t = np.tanh((field+np.sqrt(q)*z)/temperature)
            f = t if kind == 'm' else t*t if kind == 'q' else (1-t*t)**2
            return np.exp(-z*z/2)/np.sqrt(2*np.pi)*f
        centre = -field/np.sqrt(q) if q > 0 else 0.
        return quad(integrand, -12, 12, points=[centre] if -12 < centre < 12 else [],
                    epsabs=2e-11, epsrel=2e-11, limit=200)[0]
    return average('m'), average('q'), average('replicon')


def overlap(temperature, field):
    return brentq(lambda q: q-moments(temperature, field, q)[1],
                  1e-10, 1., xtol=1e-12)


def main():
    temperatures = np.linspace(.02, .995, 120)
    rows = []
    for temperature in temperatures:
        q_sg = overlap(temperature, 0.)
        onset = temperature/(1-q_sg)
        def replicon(field):
            q = overlap(temperature, field)
            return 1-moments(temperature, field, q)[2]/temperature**2
        field = brentq(replicon, 0., 5., xtol=1e-11)
        q = overlap(temperature, field)
        m, q_rhs, integral = moments(temperature, field, q)
        j0_at = field/m
        assert max(abs(q-q_rhs), abs(1-integral/temperature**2)) < 1e-8
        assert abs(m-moments(temperature, j0_at*m, q)[0]) < 1e-10
        assert j0_at > onset
        rows.append((temperature, onset, j0_at, m, q))
    data = np.asarray(rows)
    # The two curves meet at the analytic multicritical point (J0/J1,T/J1)=(1,1).
    t = np.r_[data[:, 0], 1.]
    onset = np.r_[data[:, 1], 1.]
    at = np.r_[data[:, 2], 1.]
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.4, 5.8), layout='constrained')
    ax.fill_betweenx([0, 1], 0, 1, color='#eee0ed')
    ax.fill_betweenx(t, 1, at, color='#f5e2cb')
    ax.fill_betweenx(t, at, 2.2, color='#e2eee5')
    # The computed AT line has already left the displayed bias range at Tmin.
    # Eq. (20) of de Almeida-Thouless (1978) gives J0_AT -> infinity as T -> 0.
    assert at[0] > 2.2
    ax.fill_betweenx([0, t[0]], 1, 2.2, color='#f5e2cb')
    ax.fill_between([0, 1], 1, 2.2, color='#e5edf6')
    x = np.linspace(1, 2.2, 100)
    ax.fill_between(x, x, 2.2, color='#e5edf6')
    ax.fill_between(x, 1, x, color='#e2eee5')
    ax.plot([0, 1, 2.2], [1, 1, 2.2], color='#233b53', lw=2,
            label='Paramagnetic ordering boundaries')
    ax.plot(at, t, color='#9c3374', lw=2.3, label='AT instability of the RS solution')
    ax.plot([1, 1], [0, 1], color='#752455', lw=2,
            label='Spin glass / glassy ferromagnet (RSB)')
    ax.plot(onset, t, '--', color='#545454', lw=1.8,
            label='RS SG/FM continuation (unstable)')
    ax.plot([onset[0], np.sqrt(np.pi/2)], [t[0], 0], '--', color='#545454', lw=1.8)
    ax.plot(1, 1, 'o', color='#233b53', ms=5)
    ax.text(.45, 1.65, 'Paramagnet\n$m=q=0$', ha='center', va='center')
    ax.text(1.85, 1.22, 'Ferromagnet\nRS stable', ha='center', va='center')
    ax.text(.48, .48, 'Spin glass\n$m=0$, RSB', ha='center', va='center',
            color='#752455')
    ax.text(1.5, .11, 'Glassy ferromagnet\n$m\\ne0$, RSB', ha='center', va='center',
            color='#84521d', fontsize=10.5)
    ax.set(xlim=(0, 2.2), ylim=(0, 2.2), xlabel=r'Ferromagnetic bias $J_0/J_1$',
           ylabel=r'Temperature $T/J_1$',
           title='Symmetric SK model, zero external field\nEquilibrium phases and RS stability')
    ax.legend(loc='upper left', fontsize=9, framealpha=.95)
    project = Path(__file__).resolve().parents[1]
    figure_dir, output_dir = project/'figures', project/'Output'
    figure_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir/'sk_rs_phase_diagram.png', dpi=200)
    np.savetxt(output_dir/'sk_rs_phase_boundaries.csv', data, delimiter=',',
               header='T_over_J1,RS_SG_FM_J0_over_J1,AT_J0_over_J1,AT_m,AT_q', comments='')
    print(f'Generated phase diagram; verified {len(rows)} AT and RS saddle points.')


if __name__ == '__main__':
    main()
