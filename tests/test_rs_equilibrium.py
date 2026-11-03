import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from rs_equilibrium import rs_equilibrium, add_equilibrium_references


def test_paramagnetic_reference():
    result = rs_equilibrium(2., .2, 1.3, 0., .3)
    assert result['m'] == 0 and result['q'] == 0
    assert result['energy'] == pytest.approx(-1.3**2/4)


@pytest.mark.parametrize('temperature', [.01, .5, 1.05])
def test_broken_symmetry_and_independent_equation_and_energy_checks(temperature):
    positive = rs_equilibrium(temperature,1.5,1.,0.,.3)
    negative = rs_equilibrium(temperature,1.5,1.,0.,-.2)
    m,q = positive['m'],positive['q']
    assert m > 0
    assert negative['m'] == -m
    assert negative['q'] == q and negative['energy'] == positive['energy']
    mean,sigma = 1.5*m, math.sqrt(q)
    def gaussian(z): return math.exp(-z*z/2)/math.sqrt(2*math.pi)
    centre = -mean/sigma
    points = [centre-20*temperature/sigma,centre,centre+20*temperature/sigma]
    def integrate(f):
        return quad(lambda z: gaussian(z)*f(mean+sigma*z),-12,12,
                    points=[p for p in points if -12<p<12],epsabs=1e-10)[0]
    assert integrate(lambda a: math.tanh(a/temperature)) == pytest.approx(m,abs=1e-9)
    assert integrate(lambda a: math.tanh(a/temperature)**2) == pytest.approx(q,abs=1e-9)
    # Direct conditional first moment of distribution (6.20), without using
    # integration by parts or the closed-form energy expression.
    energy = -.5*(integrate(lambda a:a*math.tanh(a/temperature))+(1-q)/temperature)
    assert energy == pytest.approx(positive['energy'],abs=1e-8)


def test_zero_initial_magnetization_selects_symmetric_spin_glass_branch():
    result = rs_equilibrium(.5,1.5,1.,0.,0.)
    assert result['m'] == 0 and result['q'] > .5
    assert rs_equilibrium(.5,0.,1.,0.,.3)['m'] == 0


def test_zero_temperature_limit():
    zero = rs_equilibrium(0.,2.,1.,0.,.3)
    low = rs_equilibrium(.0001,2.,1.,0.,.3)
    assert zero['q'] == 1
    assert low['m'] == pytest.approx(zero['m'],abs=1e-5)
    assert low['energy'] == pytest.approx(zero['energy'],abs=1e-4)
    assert rs_equilibrium(0.,0.,1.,0.,.2)['energy'] == pytest.approx(-math.sqrt(2/math.pi))


def test_asymmetric_reference_rejected():
    with pytest.raises(ValueError, match='only to k=0'):
        rs_equilibrium(1.,1.,1.,.5,.3)


def test_group_reference_lines_and_coincident_energy(tmp_path):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    from combined_runs import plot_comparisons
    cases = [dict(index=i,physical=dict(temperature=.5,j0=1.5,j1=1.,asymmetry=0.,m0=m0),
                  methods={}) for i,m0 in enumerate([-.3,.3,0.],1)]
    fig, = plot_comparisons(dict(cases=cases),show=False,figures_dir=tmp_path)
    references = [[line for line in ax.lines if line.get_label().startswith('RS equilibrium')]
                  for ax in fig.axes]
    assert len(references[0]) == 1 and len(references[1]) == 1
    assert all(line.get_linestyle() == ':' for panel in references for line in panel)
    assert references[0][0].get_ydata()[0] > 0
    assert 'positive branch' in references[0][0].get_label()
    assert 'positive branch' in references[1][0].get_label()
    assert (tmp_path/'cases_0001_0002_0003.png').exists()
    plt.close(fig)


def test_asymmetric_note_no_reference():
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2)
    add_equilibrium_references(*axes,[dict(temperature=1.,j0=1.,j1=1.,asymmetry=1.,m0=.3)])
    assert not axes[0].lines and not axes[1].lines
    assert 'k=0 only' in axes[0].texts[0].get_text()
    plt.close(fig)
