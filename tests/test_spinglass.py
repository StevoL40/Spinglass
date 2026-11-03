import numpy as np
import json
import pytest
from dataclasses import asdict
from spinglass import Parameters, SpinGlass, main, read_config, reserve_output
from datetime import datetime
from pathlib import Path


def test_initial_mass_and_magnetization():
    s=SpinGlass(Parameters(cells=40,gaussian_order=4))
    rho=s.initial(.3)
    assert np.isclose(s.dh*rho.sum(),1)
    assert np.isclose(s.dh*(rho[0]-rho[1]).sum(),.3)


def test_measure_conditional_normalization():
    s=SpinGlass(Parameters(cells=24,gaussian_order=4))
    rho=s.initial(.2)
    measure=s._measure(.2,.04)
    assert np.allclose(s.dh*measure.sum(axis=(0,3)),1,atol=1e-12)


def test_flux_conservation_and_spin_flip():
    s=SpinGlass(Parameters(cells=24,gaussian_order=4))
    rho=s.initial(.2)
    phi3=np.zeros_like(rho)
    new,dt=s.advance(rho,(0.,np.zeros(2),phi3,.1,np.tanh(s.h)),.0001)
    assert dt>0
    assert np.isclose(s.dh*new.sum(),1,atol=1e-12)
    assert np.min(new)>=0


def test_short_trajectory():
    s=SpinGlass(Parameters(cells=24,gaussian_order=4))
    records,rho=s.run(t_end=.001,magnetization=.2)
    assert len(records)>0
    assert np.isclose(s.dh*rho.sum(),1)


def test_config_comments_and_types(tmp_path):
    config = tmp_path / 'parameters.txt'
    config.write_text('# Example\n\ntemperature = 2.0 # comment\ncells = 24\n')
    assert read_config(config) == dict(temperature=2.0, cells=24)


@pytest.mark.parametrize('text', [
    'unknown = 1', 'cells 24', 'cells = 2.4', 'temperature = nan',
    'cells = 24\ncells = 32', 'temperature = two',
])
def test_config_rejects_invalid_entries(tmp_path, text):
    config = tmp_path / 'parameters.txt'
    config.write_text(text)
    with pytest.raises(ValueError, match=str(config)):
        read_config(config)


def test_config_run_matches_direct_run_and_records_effective_parameters(tmp_path):
    config = tmp_path / 'parameters.txt'
    config.write_text('cells = 24\ngaussian_order = 6\nt_end = .001\nm0 = .3\nj1 = 1.1\n')
    output = tmp_path / 'result.npz'
    main(['--config', str(config), '--quadrature', '4', '--output', str(output)])
    parameters = Parameters(cells=24, gaussian_order=4, j1=1.1)
    model = SpinGlass(parameters)
    expected, rho = model.run(.001, .3)
    with np.load(next(tmp_path.glob(output.stem + '_*.npz')), allow_pickle=False) as saved:
        assert np.array_equal(saved['trajectory'], expected)
        assert np.array_equal(saved['rho'], rho)
        assert np.array_equal(saved['h'], model.h)
        metadata = json.loads(saved['parameters'].item())
        solver = json.loads(saved['solver'].item())
        assert solver['spatial_flux'] == 'scharfetter_gummel_v1'
        assert solver['transport_timestep'] == '0.8_over_max_outgoing_rate'
    assert Path(metadata.pop('output_file')).exists()
    assert datetime.fromisoformat(metadata.pop('run_started')).tzinfo is not None
    assert metadata.pop('transport_normalization') == 'fitted_saddle_density'
    assert metadata.pop('transport_density_floor') == 1e-12
    assert metadata.pop('svd_recoveries') == 0
    assert metadata.pop('max_saddle_evaluations') == 0
    assert metadata.pop('run_status') == 'complete'
    assert metadata.pop('completed_time') == .001
    assert metadata.pop('failure_message') is None
    assert metadata == asdict(parameters) | dict(
        t_end=.001, m0=.3, quadrature=4, output=str(output), config=str(config))


def test_legacy_cli_defaults_and_metadata(tmp_path):
    output = tmp_path / 'result.npz'
    main(['--t-end', '0', '--output', str(output)])
    with np.load(next(tmp_path.glob('result_*.npz'))) as saved:
        metadata = json.loads(saved['parameters'].item())
        assert saved['rho'].shape == (2, 32)
    assert metadata['gaussian_order'] == metadata['quadrature'] == 8
    assert metadata['temperature'] == 1.5
    assert metadata['m0'] == .2


@pytest.mark.parametrize('text', ['dt_max = 0', 'saddle_maxiter = 0',
                                  'saddle_damping = 2', 't_end = -1', 'm0 = 2'])
def test_invalid_run_settings_fail_before_simulation(tmp_path, text):
    config = tmp_path / 'parameters.txt'
    config.write_text(text)
    with pytest.raises(SystemExit) as error:
        main(['--config', str(config)])
    assert error.value.code == 2


def test_single_argument_run_with_output_in_file(tmp_path):
    output = tmp_path / 'my trajectory.npz'
    config = tmp_path / 'parameters.txt'
    config.write_text(f't_end = .001\ncells = 24\ngaussian_order = 4\noutput = {output}\n')
    main([str(config)])
    with np.load(next(tmp_path.glob(output.stem + '_*.npz')), allow_pickle=False) as saved:
        metadata = json.loads(saved['parameters'].item())
        assert len(saved['trajectory']) > 0
    assert metadata['output'] == str(output)
    assert metadata['config'] == str(config)


def test_cli_output_overrides_file(tmp_path):
    unused = tmp_path / 'unused.npz'
    output = tmp_path / 'override.npz'
    config = tmp_path / 'parameters.txt'
    config.write_text(f't_end = 0\noutput = {unused}\n')
    main(['--config', str(config), '--output', str(output)])
    assert len(list(tmp_path.glob('override_*.npz'))) == 1
    assert not list(tmp_path.glob('unused*.npz'))


def test_empty_output_is_rejected(tmp_path):
    config = tmp_path / 'parameters.txt'
    config.write_text('output = # empty')
    with pytest.raises(ValueError, match='output must not be empty'):
        read_config(config)


def test_conflicting_config_arguments_are_rejected():
    with pytest.raises(SystemExit) as error:
        main(['one.txt', '--config', 'two.txt'])
    assert error.value.code == 2


@pytest.mark.parametrize('name', ['output', 'output.npz'])
def test_timestamped_output_never_overwrites_on_collision(tmp_path, name):
    started = datetime(2026, 10, 18, 23, 11, 12, 345678)
    first, handle = reserve_output(tmp_path / name, started)
    with handle:
        handle.write(b'original')
    second, handle = reserve_output(tmp_path / name, started)
    with handle:
        handle.write(b'new')
    assert first.name == 'output_2026-10-18_231112_345678.npz'
    assert second.name == 'output_2026-10-18_231112_345678_1.npz'
    assert first.read_bytes() == b'original'
    assert second.read_bytes() == b'new'


def test_progress_reports_during_saddle_without_changing_results(monkeypatch, capsys):
    from itertools import count
    ticks = count(0, 11)
    monkeypatch.setattr('spinglass.time.monotonic', lambda: next(ticks))
    parameters = Parameters(cells=24, gaussian_order=4)
    expected, expected_rho = SpinGlass(parameters).run(.001, .2)
    assert capsys.readouterr().out == ''
    records, rho = SpinGlass(parameters).run(.001, .2, progress_interval=10.)
    output = capsys.readouterr().out
    assert 'Starting simulation' in output
    assert 'Solving saddle for step 1, evaluation=' in output
    assert 'residual=' in output
    assert 'Simulation complete: t=0.001/0.001 (100.0%), steps=1' in output
    assert np.array_equal(records, expected)
    assert np.array_equal(rho, expected_rho)


def test_short_run_prints_start_and_completion_without_periodic_spam(monkeypatch, capsys):
    monkeypatch.setattr('spinglass.time.monotonic', lambda: 0.)
    SpinGlass(Parameters(cells=24, gaussian_order=4)).run(.001, .2, progress_interval=10.)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('Starting simulation')
    assert lines[1].startswith('Simulation complete')


def test_cli_prints_progress_and_saved_path(tmp_path, capsys):
    main(['--t-end', '0', '--output', str(tmp_path / 'result')])
    output = capsys.readouterr().out
    assert 'Starting simulation' in output
    assert 'Simulation complete' in output
    assert 'Saved' in output


def test_multistep_run_satisfies_every_saddle_equation():
    model = SpinGlass(Parameters(cells=24, gaussian_order=4))
    trajectory, rho = model.run(.01, .2)
    assert len(trajectory) == 5
    assert np.isclose(trajectory[-1, 0], .01)
    assert np.min(rho) >= 0
    assert np.isclose(model.dh*rho.sum(), 1, atol=1e-12)
    assert len(model.saddle_history) == len(trajectory)
    for row in model.saddle_history:
        assert set(row) == {'density', 'm', 'mu', 'q', 'theta0', 'theta1', 'Q0',
                            'normalization', 'covariance'}
        assert max(abs(value) for value in row.values()) <= model.p.saddle_tol
    assert np.array_equal(trajectory[:, -1],
                          [max(abs(v) for v in row.values()) for row in model.saddle_history])


def test_residual_checker_detects_each_multiplier_equation_violation():
    model = SpinGlass(Parameters(cells=24, gaussian_order=4))
    rho = model.initial(.2)
    sol = model.saddle(rho)
    for name, value in [('mu', .01), ('theta0', .01), ('theta1', .01), ('Q0', .01)]:
        residuals = model.saddle_residuals(rho, measure=sol['measure'], **{name: value})
        assert abs(residuals[name]) > model.p.saddle_tol
    residuals = model.saddle_residuals(rho, measure=sol['measure'], q=sol['q']+.01)
    assert abs(residuals['q']) > model.p.saddle_tol
    residuals = model.saddle_residuals(rho, measure=sol['measure'], m=sol['m']+.01)
    assert abs(residuals['m']) > model.p.saddle_tol


def test_known_high_temperature_equilibrium_saddle():
    # Thesis (6.19)-(6.20), m=q=Q0=R1=mu=0 and R0=beta/2.
    # The independent density is a pair of Gaussian fields centred at +/- beta*J^2.
    parameters = Parameters(cells=64, gaussian_order=8, hmin=-8, hmax=8, temperature=2.)
    model = SpinGlass(parameters)
    beta = 1/parameters.temperature
    spin = np.array([1., -1.])[:, None]
    rho = np.exp(-.5*(model.h-spin*beta)**2)/(2*np.sqrt(2*np.pi))
    rho /= model.dh*rho.sum()
    model.initial(0.)
    model.chi = np.exp(spin*beta*model.h/2)
    model.theta0 = beta/2
    sol = model.saddle(rho)
    assert max(abs(v) for v in sol['residuals'].values()) < 1e-10
    assert abs(sol['q']) < 1e-12
    assert abs(model.theta0-beta/2) < 1e-10
    phi1, phi2, phi3, phi4, th = model.coefficients(rho, sol)
    assert np.isclose(phi1/phi4, 1.)
    assert np.allclose(phi2/phi4, [-beta, beta])
    assert np.allclose(phi3, 0.)


@pytest.mark.parametrize('cells, temperature, j1', [
    (32, 1.5, 1.), (128, 1.5, 1.), (64, 2., 1.3),
])
def test_transport_preserves_analytic_sk_equilibrium(cells, temperature, j1):
    model = SpinGlass(Parameters(cells=cells, gaussian_order=4, hmin=-12,
                                hmax=12, j0=.75, j1=j1, temperature=temperature,
                                saddle_tol=1e-10))
    model.initial(0.)
    beta = 1/temperature
    spin = np.array([1., -1.])[:, None]
    rho = np.exp(-.5*((model.h-spin*beta*j1*j1)/j1)**2)/(2*np.sqrt(2*np.pi)*j1)
    rho /= model.dh*rho.sum()
    model.chi = np.exp(spin*beta*model.h/2)
    model.theta0 = beta/2
    sol = model.saddle(rho)
    coeff = model.coefficients(rho, sol)
    initial = rho.copy()
    for _ in range(100):
        rho, dt = model.advance(rho, coeff, .002)
    np.testing.assert_allclose(rho, initial, atol=2e-14, rtol=2e-12)
    assert model.energy(rho) == pytest.approx(-j1*j1/(2*temperature), abs=2e-13)


@pytest.mark.parametrize('diffusion', [0., 1e-300, .2, 10.])
def test_exponential_transport_rates_and_cfl_extremes(diffusion):
    model = SpinGlass(Parameters(cells=16, dt_max=100.))
    velocity = np.array([-1e4, -1., -1e-10, 0., 1e-10, 1., 1e4])
    with np.errstate(all='raise', under='ignore'):
        right, left = model._face_transport_rates(velocity, diffusion)
    assert np.all(np.isfinite(right)) and np.all(np.isfinite(left))
    assert np.min(right) >= 0 and np.min(left) >= 0
    np.testing.assert_allclose(right-left, velocity, atol=1e-14)
    rho = model.initial(.3)
    # Strong converging drift stresses the sum of the two outgoing rates.
    coeff = (1e4, np.zeros(2), np.zeros_like(rho), diffusion, np.tanh(model.h))
    evolved, dt = model.advance(rho, coeff, 100.)
    assert 0 < dt < 100.
    assert evolved.min() >= 0
    assert model.dh*evolved.sum() == pytest.approx(1., abs=1e-14)
    if diffusion == 0:
        np.testing.assert_array_equal(right, np.maximum(velocity, 0.))
        np.testing.assert_array_equal(left, np.maximum(-velocity, 0.))


def test_exponential_transport_local_zero_current():
    model = SpinGlass(Parameters(cells=32))
    peclet = np.array([-20., -1., -1e-6, 0., 1e-6, 1., 20.])
    diffusion = .7
    right, left = model._face_transport_rates(peclet*diffusion/model.dh, diffusion)
    np.testing.assert_allclose(right, left*np.exp(peclet), rtol=2e-14, atol=1e-14)


def test_high_temperature_trajectory_reaches_sk_energy():
    # Full self-consistent evolution from a random spin state, not a frozen
    # equilibrium-coefficient test. The old 32-cell upwind flux misses E=-1/3.
    model = SpinGlass(Parameters(j0=.75, temperature=1.5, cells=32,
                                hmin=-8, hmax=8, gaussian_order=2,
                                dt_max=.01, saddle_tol=1e-9))
    trajectory, rho = model.run(20., 0.)
    assert trajectory[-1, 0] == pytest.approx(20.)
    assert model.energy(rho) == pytest.approx(-1/3, abs=2e-6)
    assert rho.min() >= 0
    assert model.dh*rho.sum() == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize('magnetization, settings', [
    (0., {}), (-.2, {}), (.2, {'temperature': 0.}),
    (.2, {'j0': .5, 'j1': 1.2, 'asymmetry': .3}),
])
def test_strict_solver_varied_physical_settings(magnetization, settings):
    model = SpinGlass(Parameters(cells=24, gaussian_order=4, **settings))
    trajectory, rho = model.run(.01, magnetization)
    assert np.isclose(trajectory[-1, 0], .01)
    assert np.min(rho) >= 0
    for row in model.saddle_history:
        assert max(abs(v) for v in row.values()) <= model.p.saddle_tol


def test_unconverged_fit_does_not_commit_state():
    model = SpinGlass(Parameters(cells=24, gaussian_order=4, saddle_maxiter=1))
    rho = model.initial(.2)
    sol = model.saddle(rho)
    evolved, dt = model.advance(rho, model.coefficients(rho, sol), .002)
    before = (model.q, model.mu, model.theta0, model.theta1, model.r, model.chi.copy())
    with pytest.raises(RuntimeError, match='did not satisfy all equations'):
        model.saddle(evolved)
    assert (model.q, model.mu, model.theta0, model.theta1, model.r) == before[:5]
    assert np.array_equal(model.chi, before[5])


def test_saved_equation_diagnostics(tmp_path):
    main(['--cells', '24', '--quadrature', '4', '--t-end', '.01',
          '--output', str(tmp_path/'strict.npz')])
    with np.load(next(tmp_path.glob('strict_*.npz')), allow_pickle=False) as data:
        residuals = data['saddle_residuals']
        assert residuals.shape == (5, 9)
        assert set(data['saddle_residual_names']) == {'density', 'm', 'mu', 'q', 'theta0',
                                                    'theta1', 'Q0', 'normalization', 'covariance'}
        assert np.max(np.abs(residuals)) <= Parameters().saddle_tol
        assert np.array_equal(data['trajectory'][:, -1], np.max(np.abs(residuals), axis=1))
        assert np.allclose(data['saddle_times'], [0, .002, .004, .006, .008])
        assert json.loads(data['solver'].item())['method'] == 'bounded_least_squares_RS_v3'


def test_invalid_covariance_is_rejected_without_fallback():
    model = SpinGlass(Parameters(cells=24, gaussian_order=4))
    model.initial(.2)
    model.theta1 = .1
    with pytest.raises(FloatingPointError, match='covariance'):
        model._measure(.2, .04)


def test_higher_magnetization_needs_resolved_quadrature():
    coarse = SpinGlass(Parameters(cells=24, gaussian_order=4))
    with pytest.raises(RuntimeError, match='worst=Q0'):
        coarse.saddle(coarse.initial(.6))
    resolved = SpinGlass(Parameters(cells=24, gaussian_order=8, temperature=0.))
    trajectory, rho = resolved.run(.01, .6)
    assert np.isclose(trajectory[-1, 0], .01)
    assert np.max(trajectory[:, -1]) <= resolved.p.saddle_tol


@pytest.mark.parametrize("asymmetry", [.3, 1.])
def test_logweight_jacobian_against_independent_finite_differences(monkeypatch, asymmetry):
    from scipy.optimize import least_squares as scipy_least_squares
    checked = []
    def checking_fit(fun, x0, jac, **kwargs):
        analytic = jac(x0)
        # Check logweight columns, independently of the solver's three numerical columns.
        numerical = np.zeros_like(analytic[:, :-3])
        for j in range(len(x0)-3):
            plus = x0.copy(); minus = x0.copy()
            plus[j] += 1e-5; minus[j] -= 1e-5
            numerical[:, j] = (fun(plus)-fun(minus))/(2e-5)
        assert np.allclose(analytic[:, :-3], numerical, atol=1e-8, rtol=1e-5)
        checked.append(True)
        return scipy_least_squares(fun, x0, jac=jac, **kwargs)
    monkeypatch.setattr('spinglass.least_squares', checking_fit)
    model = SpinGlass(Parameters(cells=24, gaussian_order=4, j0=.4, j1=1.2, asymmetry=asymmetry))
    model.run(.004, .2)
    assert checked


def test_zero_step_output_has_empty_diagnostics(tmp_path):
    main(['--t-end', '0', '--output', str(tmp_path/'empty')])
    with np.load(next(tmp_path.glob('empty_*.npz'))) as data:
        assert data['saddle_residuals'].shape == (0, 9)
        assert data['saddle_times'].shape == (0,)


def test_rotated_covariance_measure_matches_original_gaussian_integral():
    from scipy.special import logsumexp
    model = SpinGlass(Parameters(cells=24, gaussian_order=32))
    model.initial(.2)
    model.q = .15
    model.r = .4
    model.theta0 = .07
    model.theta1 = .03
    model.mu = .02
    q = model.q
    x = model.h
    spin = np.array([1., -1.])[:, None, None, None]
    ratio = model.ok*model.theta1/model.r
    lam = np.sqrt(q)/(model.p.j1*(1-q))
    original_logw = (np.log(model.chi)[:, None, None, :]
                     -x*x/(2*model.js*(1-q))
                     +spin*(model.ok*model.theta0*x+model.mu)
                     +(ratio*x+spin*model.r)*model.z2[None, :, :, None]
                     +np.sqrt(lam*lam-ratio*ratio)*x*model.z1[None, :, :, None])
    original = np.exp(original_logw-logsumexp(original_logw, axis=(0, 3), keepdims=True)
                      -np.log(model.dh))
    reference = model._observables(original, .2)
    rotated = model._observables(model._measure(.2, q), .2)
    for actual, expected in zip(rotated, reference):
        assert np.allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_optimizer_success_flag_cannot_accept_bad_equations(monkeypatch):
    from types import SimpleNamespace
    model = SpinGlass(Parameters(cells=24, gaussian_order=4))
    rho = model.initial(.2)
    sol = model.saddle(rho)
    evolved, _ = model.advance(rho, model.coefficients(rho, sol), .002)
    monkeypatch.setattr('spinglass.least_squares',
                        lambda fun, x0, **kwargs: SimpleNamespace(x=x0, nfev=1, success=True))
    with pytest.raises(RuntimeError, match='did not satisfy all equations'):
        model.saddle(evolved)


def test_biased_initial_gaussian_with_nonunit_coupling():
    model = SpinGlass(Parameters(cells=64, gaussian_order=12, hmin=-10, hmax=10,
                                j0=.5, j1=1.2))
    rho = model.initial(.2)
    assert model.rterm == model.js*.2**2
    residuals = model.saddle_residuals(rho)
    assert max(abs(v) for v in residuals.values()) < 1e-9


def test_final_density_can_be_closed_without_fallbacks():
    model = SpinGlass(Parameters(cells=24, gaussian_order=4))
    trajectory, rho = model.run(.1, .2)
    sol = model.saddle(rho)
    independently_recomputed = model.saddle_residuals(rho)
    assert len(trajectory) == 50
    assert max(abs(v) for v in independently_recomputed.values()) <= model.p.saddle_tol
    assert np.allclose(sol['measure'], model._measure(sol['m'], sol['q']), atol=1e-12)


def load_late_time_checkpoint(order=4, saddle_tol=2e-5):
    model = SpinGlass(Parameters(cells=24, gaussian_order=order, saddle_tol=saddle_tol))
    model.initial(.2)
    with np.load(Path(__file__).parent/'saddle_t6_036.npz', allow_pickle=False) as data:
        rho = data['rho'].copy()
        for name in ('chi', 'q', 'mu', 'theta0', 'theta1', 'r', 'rterm', 'tterm'):
            setattr(model, name, data[name].copy() if name == 'chi' else data[name].item())
    return model, rho


def test_reported_late_time_failure_and_objective_consistency():
    model, rho = load_late_time_checkpoint()
    solution = model.saddle(rho)
    residuals = model.saddle_residuals(rho)
    assert max(abs(v) for v in residuals.values()) <= model.p.saddle_tol
    assert solution['residual'] <= model.p.saddle_tol
    names = ('m', 'mu', 'q', 'theta0', 'theta1', 'Q0')
    # The optimizer must now see the same scalar equations as the independent check.
    assert np.allclose(solution['objective_residual'][-6:], [residuals[name] for name in names], atol=1e-12)
    assert np.max(np.abs(solution['objective_residual'][:-6])) == pytest.approx(residuals['density'], abs=1e-12)


@pytest.mark.parametrize('order', [8, 12])
def test_late_time_density_with_resolved_quadrature(order):
    model, rho = load_late_time_checkpoint(order, saddle_tol=1e-11)
    solution = model.saddle(rho)
    assert solution['residual'] < 1e-10
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) < 1e-10


def test_late_time_objective_jacobian(monkeypatch):
    from scipy.optimize import least_squares as scipy_least_squares
    checked = []
    def checking_fit(fun, x0, jac, **kwargs):
        actual = jac(x0)
        expected = np.zeros_like(actual[:, :-3])
        for j in range(len(x0)-3):
            plus = x0.copy(); minus = x0.copy()
            plus[j] += 1e-5; minus[j] -= 1e-5
            expected[:, j] = (fun(plus)-fun(minus))/(2e-5)
        assert np.allclose(actual[:, :-3], expected, atol=1e-8, rtol=1e-5)
        checked.append(True)
        return scipy_least_squares(fun, x0, jac=jac, **kwargs)
    monkeypatch.setattr('spinglass.least_squares', checking_fit)
    # Resolve the quadrature and tighten tolerance so numerical Jacobian probes
    # cannot trigger the solver's legitimate early-acceptance exception.
    model, rho = load_late_time_checkpoint(order=8, saddle_tol=1e-11)
    model.saddle(rho)
    assert checked


def test_initial_energy_includes_mean_coupling():
    model = SpinGlass(Parameters(cells=64, gaussian_order=8, j0=.5, hmin=-8, hmax=8))
    rho = model.initial(.3)
    assert model.energy(rho) == pytest.approx(-.5*.5*.3**2, abs=1e-12)


def test_saved_energy_matches_every_updated_density(tmp_path, monkeypatch):
    expected = []
    original = SpinGlass.advance
    def capture(self, rho, coeff, dt):
        updated, step = original(self, rho, coeff, dt)
        expected.append(-.5*self.dh*np.sum(self.h*(updated[0]-updated[1])))
        return updated, step
    monkeypatch.setattr(SpinGlass, 'advance', capture)
    main(['--cells', '24', '--quadrature', '8', '--t-end', '.01',
          '--output', str(tmp_path/'energy')])
    with np.load(next(tmp_path.glob('energy_*.npz'))) as data:
        assert data['energy'].shape == (len(data['trajectory']),)
        assert np.allclose(data['energy'], expected, atol=1e-14)
        assert data['energy'][-1] == pytest.approx(
            -.5*(data['h'][1]-data['h'][0])*np.dot(data['h'],data['rho'][0]-data['rho'][1]))


def test_logweights_preserve_measure_when_linear_weights_underflow():
    model = SpinGlass(Parameters(cells=24, gaussian_order=8))
    rho = model.initial(.3)
    reference = model._measure(.3, model.q)
    model.logchi -= 1000.  # Same normalized measure, but every linear weight underflows.
    assert np.all(model.chi == 0)
    assert np.all(np.isfinite(model.logchi))
    assert np.allclose(model._measure(.3, model.q), reference, atol=1e-13)
    assert model.saddle(rho)['residual'] <= model.p.saddle_tol


def test_low_temperature_underflow_checkpoint(monkeypatch):
    import warnings
    from scipy.optimize import least_squares as scipy_least_squares
    model = SpinGlass(Parameters(cells=24, gaussian_order=8, temperature=.01))
    model.initial(.3)
    with np.load(Path(__file__).parent/'saddle_low_temperature_t2_466.npz') as data:
        rho = data['rho'].copy()
        model.logchi = data['logchi'].copy()
        for name in ('q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model,name,data[name].item())
    assert np.any((model.chi == 0)&(rho > 0))
    assert np.all(np.isfinite(model.logchi))
    fits = []
    def finite_fit(fun, x0, **kwargs):
        assert np.all(np.isfinite(x0))
        fits.append(x0.copy())
        return scipy_least_squares(fun,x0,**kwargs)
    monkeypatch.setattr('spinglass.least_squares', finite_fit)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        solution = model.saddle(rho)
    assert fits
    assert solution['residual'] <= model.p.saddle_tol
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


def test_zero_weight_on_positive_density_has_clear_error(monkeypatch):
    model = SpinGlass(Parameters(cells=24, gaussian_order=8))
    rho = model.initial(.3)
    chi = model.chi
    chi[0,0] = 0.
    model.chi = chi  # A genuinely supplied zero has no finite logarithm to retain.
    def must_not_optimize(*args, **kwargs):
        raise AssertionError('Nonfinite parameters must not reach SciPy')
    monkeypatch.setattr('spinglass.least_squares', must_not_optimize)
    with pytest.raises(FloatingPointError,match='Nonfinite log weight'):
        model.saddle(rho)


@pytest.mark.parametrize('order', [8,12,16])
def test_later_low_temperature_state_retains_resolution_check(order):
    model = SpinGlass(Parameters(cells=24,gaussian_order=order,temperature=.01,
                                saddle_tol=2e-5 if order==8 else 1e-10))
    model.initial(.3)
    with np.load(Path(__file__).parent/'saddle_low_temperature_t5_432.npz') as data:
        rho = data['rho'].copy()
        for name in ('chi','q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model,name,data[name].copy() if name=='chi' else data[name].item())
    if order == 8:
        with pytest.raises(RuntimeError,match='worst=theta0'):
            model.saddle(rho)
    else:
        solution = model.saddle(rho)
        assert solution['residual'] < 1e-9


def test_density_matching_seed_recovers_a_flat_tail_direction(monkeypatch):
    from scipy.optimize import least_squares as scipy_least_squares
    model = SpinGlass(Parameters(cells=24,gaussian_order=8))
    rho = model.initial(.3)
    sol = model.saddle(rho)
    rho, _ = model.advance(rho,model.coefficients(rho,sol),.002)
    model.logchi[0,0] = -1e12
    assert model.chi[0,0] == 0 and rho[0,0] > 0
    fits = []
    def checking_fit(fun,x0,**kwargs):
        assert np.all(np.isfinite(x0))
        assert np.min(x0[:-3]) > -1000  # The density-based seed recovers the lost direction.
        fits.append(True)
        return scipy_least_squares(fun,x0,**kwargs)
    monkeypatch.setattr('spinglass.least_squares',checking_fit)
    solution = model.saddle(rho)
    assert fits
    assert solution['residual'] <= model.p.saddle_tol


def test_optimizer_stops_on_verified_equations_without_oversolving(monkeypatch):
    from scipy.optimize import least_squares as scipy_least_squares
    calls = []
    returned_normally = []
    def fit_with_counter(fun,x0,**kwargs):
        def counted(z):
            calls.append(z.copy())
            return fun(z)
        fit = scipy_least_squares(counted,x0,**kwargs)
        returned_normally.append(True)
        return fit
    monkeypatch.setattr('spinglass.least_squares',fit_with_counter)
    model = SpinGlass(Parameters(cells=24,gaussian_order=8))
    rho = model.initial(.3)
    initial = model.saddle(rho)
    rho,_ = model.advance(rho,model.coefficients(rho,initial),.002)
    solution = model.saddle(rho)
    assert calls and not returned_normally
    assert solution['iterations'] == len(calls)
    assert solution['residual'] <= model.p.saddle_tol
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


@pytest.mark.parametrize('k', [-.01, 1.01, np.nan, np.inf])
def test_asymmetry_project_range(k):
    from simulate_spinglass import Parameters as SimulationParameters
    for parameter_class in (Parameters, SimulationParameters):
        with pytest.raises(ValueError):
            parameter_class(asymmetry=k)


@pytest.mark.parametrize('temperature', [0., .01, 1.5])
def test_fully_asymmetric_endpoint_preserves_equations_and_magnetization(temperature):
    model = SpinGlass(Parameters(cells=24, gaussian_order=12, asymmetry=1., temperature=temperature))
    trajectory, rho = model.run(.1, .3)
    assert np.isfinite(trajectory).all() and np.isfinite(rho).all()
    assert np.max(trajectory[:, -1]) <= model.p.saddle_tol
    assert abs(model.dh*rho.sum()-1) < 1e-12 and np.min(rho) >= 0
    # J0=0 and an even total field distribution imply dm/dt=-m at k=1.
    np.testing.assert_allclose(trajectory[:, 1], .3*np.exp(-trajectory[:, 0]), atol=2e-10)
    solution = model.saddle(rho)
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol
    assert np.isfinite(model.theta1)
    # Independently reconstruct the committed measure: cross-covariance is zero.
    np.testing.assert_allclose(model._measure(solution['m'], solution['q']), solution['measure'], atol=1e-12)


def _ferromagnetic_checkpoint(k, budget=120):
    suffix = 'k0_t5_115645' if k == 0 else 'k1_t6_805276'
    with np.load(Path(__file__).parent/f'saddle_ferromagnetic_{suffix}.npz') as data:
        settings = json.loads(data['parameters'].item())
        settings['saddle_maxiter'] = budget
        model = SpinGlass(Parameters(**settings))
        rho = data['rho'].copy()
        model.logchi = data['logchi'].copy()
        for name in ('q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model, name, data[name].item())
    return model, rho


@pytest.mark.parametrize('k', [0, 1])
def test_reported_ferromagnetic_states_close_with_original_equation_checks(k):
    model, rho = _ferromagnetic_checkpoint(k)
    sol = model.saddle(rho)
    assert sol['iterations'] <= model.p.saddle_maxiter
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


def test_conditional_transport_does_not_amplify_accepted_tail_density_error():
    # A real failing-run state can be closed with a larger budget, but its small
    # absolute tail error is enough to create a false, very restrictive CFL bound.
    # Freeze the historical accepted approximation so improvements to the
    # initializer cannot remove the tail-error scenario this test exercises.
    with np.load(Path(__file__).parent/'saddle_ferromagnetic_k1_accepted_tail_error.npz') as data:
        model=SpinGlass(Parameters(**json.loads(data['parameters'].item())))
        rho=data['rho'].copy();model.logchi=data['logchi'].copy()
        for name in ('q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model,name,data[name].item())
    magnetization=model.dh*(rho[0]-rho[1]).sum()
    measure=model._measure(magnetization,model.q)
    target,sr,hr,*_=model._observables(measure,magnetization)
    sol=dict(m=magnetization,q=model.q,measure=measure,sr=sr,hr=hr)
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol
    fitted = np.einsum('ij,sijh->sh', model.gw, sol['measure'])
    phi1, phi2, phi3, phi4, th = model.coefficients(rho, sol)
    x = model.h-model.p.j0*sol['m']
    q = sol['q']; l0 = (1-2*q)/(model.js*(1-q)**2); l1 = q/(model.js*(1-q)**2)
    tp = model.dh*np.dot(th,rho[0]); tm = model.dh*np.dot(th,rho[1])
    ts = model.dh*np.dot(th,sol['sr'].sum(axis=0))
    # k=1: ko=0, leaving just this original unnormalized Phi3 term.
    original_phi3 = model.js*sol['hr']*((l0+2*l1)*ts-l1*(tp-tm))
    np.testing.assert_allclose(phi3, rho*original_phi3/fitted, atol=1e-14)
    old_adv = -(phi1*x[None,:]+phi2[:,None]+original_phi3/np.maximum(rho,1e-12))
    old_v = .5*(old_adv[:,:-1]+old_adv[:,1:])
    new_adv = -(phi1*x[None,:]+phi2[:,None]+phi3/np.maximum(rho,1e-12))
    new_v = .5*(new_adv[:,:-1]+new_adv[:,1:])
    assert np.max(np.abs(old_v)) > 1000
    assert np.max(np.abs(new_v)) < 20
    evolved, dt = model.advance(rho,(phi1,phi2,phi3,phi4,th),model.p.dt_max)
    assert dt == model.p.dt_max
    assert np.min(evolved) >= 0 and model.dh*evolved.sum() == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize('k', [0., .5, 1.])
def test_conditional_transport_equals_original_flux_when_density_matches(k):
    model = SpinGlass(Parameters(cells=24,gaussian_order=12,asymmetry=k,j0=.4))
    rho = model.initial(.3)
    sol = model.saddle(rho)
    rho = np.einsum('ij,sijh->sh', model.gw, sol['measure'])
    q,m = sol['q'],sol['m']; x = model.h-model.p.j0*m
    th = np.tanh(model.h/model.p.temperature)
    tp=model.dh*np.dot(th,rho[0]); tm=model.dh*np.dot(th,rho[1])
    ts=model.dh*np.dot(th,sol['sr'].sum(axis=0))
    ht=model.dh*np.dot(x*th,rho.sum(axis=0)); h_t=model.dh*np.dot(th,sol['hr'].sum(axis=0))
    l0=(1-2*q)/(model.js*(1-q)**2); l1=q/(model.js*(1-q)**2)
    original=(model.js*model.ko*sol['sr']*((l0+2*l1)*h_t-l1*ht)
              +model.js*sol['hr']*((l0+2*l1)*ts-l1*(tp-tm))
              -2*model.js*model.ko*sol['sr']*((model.theta0-model.theta1)*ts+model.theta1*(tp-tm)))
    np.testing.assert_allclose(model.coefficients(rho,sol)[2],original,atol=1e-15,rtol=1e-14)


def test_svd_recovery_uses_same_solver_within_original_budget(monkeypatch):
    from scipy.optimize import least_squares as original_fit
    calls = []
    def fail_once(fun, x0, **kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            fun(x0)  # The failed solve has used one evaluation.
            raise np.linalg.LinAlgError('SVD did not converge for slice = 0.')
        assert kwargs['x_scale'] == .5
        return original_fit(fun, x0, **kwargs)
    model = SpinGlass(Parameters(cells=24,gaussian_order=8,saddle_maxiter=40))
    rho=model.initial(.3);sol=model.saddle(rho)
    rho,_=model.advance(rho,model.coefficients(rho,sol),.002)
    monkeypatch.setattr('spinglass.least_squares',fail_once)
    sol=model.saddle(rho)
    assert sol['svd_recovered']
    assert calls[0]['max_nfev']==40 and calls[1]['max_nfev']==39
    assert sol['iterations'] <= 40
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


def test_svd_recovery_does_not_create_extra_evaluation_budget(monkeypatch):
    model=SpinGlass(Parameters(cells=24,gaussian_order=8,saddle_maxiter=1))
    rho=model.initial(.3);sol=model.saddle(rho)
    rho,_=model.advance(rho,model.coefficients(rho,sol),.002)
    def fail(fun,x0,**kwargs):
        fun(x0)
        raise np.linalg.LinAlgError('SVD did not converge')
    monkeypatch.setattr('spinglass.least_squares',fail)
    with pytest.raises(RuntimeError,match='no saddle evaluation budget remains'):
        model.saddle(rho)


def test_actual_finite_jacobian_svd_regression():
    with np.load(Path(__file__).parent/'saddle_ferromagnetic_k0_svd.npz') as data:
        model=SpinGlass(Parameters(**json.loads(data['parameters'].item())))
        rho=data['rho'].copy();model.logchi=data['logchi'].copy()
        for name in ('q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model,name,data[name].item())
    sol=model.saddle(rho)
    # Some LAPACK versions need the fallback; others solve the matrix directly.
    assert sol['iterations'] <= model.p.saddle_maxiter
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


@pytest.mark.parametrize('q,b,variance', [(0.,0.,0.),(.1,.2,.3),(.9,-.7,.2),(.99,1.,.01)])
@pytest.mark.parametrize('shift', [0.,-1200.])
def test_factored_measure_matches_original_tensor_quadrature(q,b,variance,shift):
    from scipy.special import logsumexp
    model=SpinGlass(Parameters(cells=24,gaussian_order=12,j0=2.))
    logweights=np.random.default_rng(9).normal(size=(2,24))*5+shift
    logw=logweights[:,None,None,:]+model._gaussian_kernel(.3,q,b,variance)
    expected=np.exp(logw-logsumexp(logw,axis=(0,3),keepdims=True)-np.log(model.dh))
    actual=model._gaussian_measure(.3,q,logweights,b,variance)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(model.dh*actual.sum(axis=(0,3)),1.,atol=1e-14)


def test_factored_measure_supports_an_empty_spin_population():
    from scipy.special import logsumexp
    model=SpinGlass(Parameters(cells=24,gaussian_order=12))
    logweights=np.zeros((2,24));logweights[1]=-np.inf
    logw=logweights[:,None,None,:]+model._gaussian_kernel(1.,.5,.2,.3)
    expected=np.exp(logw-logsumexp(logw,axis=(0,3),keepdims=True)-np.log(model.dh))
    with np.errstate(invalid='raise',divide='raise',over='raise'):
        actual=model._gaussian_measure(1.,.5,logweights,.2,.3)
    np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-13)
    assert np.all(actual[1]==0.)


def test_repeated_matching_initialization_closes_strong_mean_late_state():
    with np.load(Path(__file__).parent/'saddle_ferromagnetic_k0_t6_516.npz') as data:
        settings=json.loads(data['parameters'].item());settings['saddle_maxiter']=120
        model=SpinGlass(Parameters(**settings));rho=data['rho'].copy()
        model.logchi=data['logchi'].copy()
        for name in ('q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model,name,data[name].item())
    sol=model.saddle(rho)
    assert sol['iterations'] < 120
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


def test_saved_saddle_evaluation_counts_match_accepted_steps(tmp_path):
    main(['--cells','24','--quadrature','8','--t-end','.01',
          '--output',str(tmp_path/'counts.npz')])
    with np.load(next(tmp_path.glob('counts_*.npz'))) as data:
        p=json.loads(data['parameters'].item())
        counts=data['saddle_iterations']
        assert counts.shape==(len(data['trajectory']),)
        assert np.issubdtype(counts.dtype,np.integer)
        assert np.all((counts>=0)&(counts<=p['saddle_maxiter']))
        assert p['max_saddle_evaluations']==max(counts,default=0)
        assert json.loads(data['solver'].item())['density_matching_steps']==4


@pytest.mark.parametrize('shift', [0.,-1400.])
def test_factored_log_density_matches_independent_log_kernel_marginal(shift):
    from scipy.special import logsumexp
    model=SpinGlass(Parameters(cells=24,gaussian_order=12,j0=.4))
    u=np.random.default_rng(12).normal(size=(2,24))*8+shift
    kernel=model._gaussian_kernel(.3,.8,-.4,.2)
    logz=logsumexp(u[:,None,None,:]+kernel,axis=(0,3),keepdims=True)+np.log(model.dh)
    expected=u+logsumexp(kernel-logz+np.log(model.gw)[None,:,:,None],axis=(1,2))
    actual=model._gaussian_log_density(.3,.8,u,-.4,.2)
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=1e-12)


def test_safeguarded_matching_preserves_better_continuation_guess():
    with np.load(Path(__file__).parent/'saddle_ferromagnetic_k0_t6_736.npz') as data:
        settings=json.loads(data['parameters'].item());settings['saddle_maxiter']=120
        model=SpinGlass(Parameters(**settings));rho=data['rho'].copy()
        model.logchi=data['logchi'].copy()
        for name in ('q','mu','theta0','theta1','r','rterm','tterm'):
            setattr(model,name,data[name].item())
    sol=model.saddle(rho)
    assert sol['iterations'] < 120
    assert max(abs(v) for v in model.saddle_residuals(rho).values()) <= model.p.saddle_tol


@pytest.mark.parametrize('completed_steps', [0, 2])
def test_saddle_failure_saves_only_completed_steps(tmp_path, monkeypatch, completed_steps):
    original = SpinGlass.saddle
    calls = 0
    last_rho = None
    def fail(self, rho, **kwargs):
        nonlocal calls, last_rho
        last_rho = rho.copy()
        if calls == completed_steps:
            raise RuntimeError('deliberate saddle failure')
        calls += 1
        return original(self, rho, **kwargs)
    monkeypatch.setattr(SpinGlass, 'saddle', fail)
    with pytest.raises(RuntimeError, match='deliberate saddle failure'):
        main(['--cells', '24', '--quadrature', '8', '--t-end', '.01',
              '--output', str(tmp_path/'partial.npz')])
    with np.load(next(tmp_path.glob('partial_*.npz')), allow_pickle=False) as data:
        metadata = json.loads(data['parameters'].item())
        assert metadata['run_status'] == 'failed'
        assert 'deliberate saddle failure' in metadata['failure_message']
        assert data['trajectory'].shape == (completed_steps, 5)
        assert data['energy'].shape == (completed_steps,)
        assert len(data['saddle_times']) == completed_steps
        assert data['saddle_residuals'].shape == (completed_steps, 9)
        np.testing.assert_array_equal(data['rho'], last_rho)
        assert metadata['completed_time'] == (data['trajectory'][-1, 0] if completed_steps else 0)


def test_combined_style_standalone_names_preserve_values(tmp_path):
    p = tmp_path/'fd.txt'
    p.write_text('T=1.5\nfd_cells=128\nfd_gaussian_order=64\nfd_dt_max=.003\n'
                 'fd_saddle_tol=1e-5\nfd_saddle_maxiter=250\noutput_dir=Output\noutput_prefix=trajectory\n')
    settings = read_config(p)
    assert settings == dict(temperature=1.5,cells=128,gaussian_order=64,dt_max=.003,
                           saddle_tol=1e-5,saddle_maxiter=250,output='Output/trajectory.npz')
    for text in ['cells=24\nfd_cells=32', 'output=a.npz\noutput_dir=Output',
                 'output_prefix=../bad', 'T=1\ntemperature=2']:
        p.write_text(text)
        with pytest.raises(ValueError): read_config(p)
