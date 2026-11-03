"""Replica-symmetric equilibrium references, thesis (6.19)-(6.20), k=0 only.

These infinite-system references do not assert dynamic convergence or RS stability.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf


@lru_cache(maxsize=1024)
def _positive_solution(temperature, j0, j1):
    """Find the symmetric and positive RS branches with adaptive Gaussian integrals."""
    t = temperature
    def averages(m, q):
        mean, sigma = j0*m, j1*math.sqrt(q)
        if sigma == 0:
            value = math.tanh(mean/t) if t else float(np.sign(mean))
            return value, value*value
        if t == 0:
            return float(erf(mean/(math.sqrt(2)*sigma))), 1.
        # Explicit breakpoints resolve the thermal transition even at low T.
        centre, width = -mean/sigma, t/sigma
        points = sorted({x for x in (centre-20*width, centre, centre+20*width)
                         if -12 < x < 12})
        def integrate(power):
            return quad(lambda z: math.exp(-z*z/2)/math.sqrt(2*math.pi)*
                        math.tanh((mean+sigma*z)/t)**power,
                        -12, 12, points=points, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
        return integrate(1), integrate(2)

    def q_at(m):
        if t == 0 and j1 > 0:
            return 1.
        if m == 0 and (j1 == 0 or t >= j1):
            return 0.
        def error(q):
            return q-averages(m, q)[1]
        # At zero magnetisation the q=0 root exists at every T. Below T=J1
        # choose the nonzero spin-glass branch rather than that trivial root.
        lo = 1e-10 if m == 0 else 0.
        if error(lo) >= 0:
            return 0.
        return brentq(error, lo, 1., xtol=1e-12)

    q_zero = q_at(0.)
    def error(m):
        return m-averages(m, q_at(m))[0]
    roots = []
    # Sign alone cannot select between multiple positive roots. Detect and
    # report such ambiguity rather than silently assigning a branch.
    grid = np.unique(np.r_[np.geomspace(1e-7, .01, 12), np.linspace(.01, 1., 65)])
    previous, f_previous = grid[0], error(grid[0])
    for current in grid[1:]:
        f_current = error(current)
        if f_previous*f_current < 0 or f_current == 0:
            root = brentq(error, previous, current, xtol=1e-11)
            if not roots or abs(root-roots[-1]) > 1e-6:
                roots.append(root)
        previous, f_previous = current, f_current
    if len(roots) > 1:
        raise ValueError('Multiple positive RS branches; initial sign alone cannot select one')
    m = roots[0] if roots else 0.
    q = q_at(m) if m else q_zero
    if max(abs(m-averages(m,q)[0]), abs(q-averages(m,q)[1])) > 1e-8:
        raise ValueError('RS equilibrium equation check failed')
    return m, q, q_zero


def rs_equilibrium(temperature, j0, j1, asymmetry, m0):
    """Select the sign of the nonzero branch from m0; exactly m0=0 uses m=0.

    At finite T, E=-[J0 m² + J1²(1-q²)/T]/2, the first spin-field
    moment of (6.20). At T=0 use its analytic limit. k>0 is not covered.
    """
    if asymmetry != 0:
        raise ValueError('Thesis RS equilibrium reference applies only to k=0')
    if not all(math.isfinite(x) for x in (temperature,j0,j1,asymmetry,m0)):
        raise ValueError('Equilibrium parameters must be finite')
    if temperature < 0 or j1 < 0 or abs(m0) > 1:
        raise ValueError('Require T>=0, J1>=0 and |m0|<=1')
    if temperature == 0 and j1 == 0:
        m = float(np.sign(m0)) if j0 > 0 else 0.
        return dict(m=m, q=m*m, energy=-.5*j0*m*m)
    magnitude, q, q_zero = _positive_solution(temperature,j0,j1)
    m = math.copysign(magnitude,m0) if m0 else 0.
    if m0 == 0:
        q = q_zero
    if temperature:
        energy = -.5*(j0*m*m + j1*j1*(1-q*q)/temperature)
    else:
        energy = -.5*j0*m*m-j1*math.sqrt(2/math.pi)*math.exp(-.5*(j0*m/j1)**2)
    return dict(m=m, q=q, energy=energy)


def add_equilibrium_references(m_panel, energy_panel, physical_sets):
    """Add positive-branch horizontal RS references, independent of initial m0."""
    selected = []
    for physical in physical_sets:
        if not all(name in physical for name in ('temperature','j0','j1','asymmetry','m0')):
            continue  # Old/incomplete metadata cannot define a physical reference.
        if physical['asymmetry'] != 0:
            m_panel.text(.02,.02,'RS equilibrium reference: k=0 only',
                         transform=m_panel.transAxes, fontsize='small')
            return
        try:
            result = rs_equilibrium(**{name:physical[name] for name in
                                      ('temperature','j0','j1','asymmetry')}, m0=1.)
        except ValueError as error:
            print(f'RS equilibrium reference unavailable: {error}')
            continue
        print(f"RS equilibrium (positive branch): m={result['m']:.8g}, "
              f"q={result['q']:.8g}, E={result['energy']:.8g}")
        selected.append((physical['m0'], result))
    for panel, observable in [(m_panel,'m'),(energy_panel,'energy')]:
        groups = []
        for m0, result in selected:
            value = result[observable]
            for group in groups:
                if math.isclose(value,group[0],abs_tol=1e-8,rel_tol=1e-8):
                    group[1].append(m0)
                    break
            else:
                groups.append((value,[m0]))
        for value, starts in groups:
            label = 'RS equilibrium '+('E' if observable=='energy' else 'm')
            label += f'={value:.4g} (positive branch)' if selected[0][1]['m'] > 0 else f'={value:.4g} (m=0 branch)'
            panel.axhline(value, color='black', linestyle=':', linewidth=1.4,label=label)
