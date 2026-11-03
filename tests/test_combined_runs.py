from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from combined_runs import (DEFAULTS, read_combined_config, combination_count,
                           physical_sets, run_sweep, load_manifest, plot_comparisons)


def config(tmp_path, text=''):
    path = tmp_path/'combined.txt'
    path.write_text(f'output_dir = {tmp_path}\n'+text)
    return path


def test_scalar_and_cartesian_product(tmp_path):
    path=config(tmp_path, 'T = [0.01,0.5,1.0,1.5,2.0]\nm0=[0.1,0.3]\n')
    s=read_combined_config(path)
    assert combination_count(s)==10
    sets=list(physical_sets(s))
    assert len(sets)==10 and sets[0]['temperature']==.01
    assert sets[0]['m0']==.1 and sets[1]['m0']==.3
    assert len({(p['temperature'],p['m0']) for p in sets})==10
    assert combination_count(read_combined_config(config(tmp_path,'T=1.0\n')))==1


@pytest.mark.parametrize('text', [
    'T=1\ntemperature=2', 'T=[]', 'T=[[1,2]]', 'T=[1,1]',
    'T=__import__("os").system("touch BAD")', 'T="1.0"', 'T=True',
    't_end=[1,2]', 'n_spins=1000.0', 'wrong=1', 'seed=-1',
    'asymmetry=1.01', 'asymmetry=-0.01', 'j1=0', 'm0=1.1', 'T=-1', 'fd_gaussian_order=1',
    'warning_threshold=0', 'output_prefix=../unsafe', 't_end=0.0001',
    'fd_dt_max=0', 'T=[1e999]',
])
def test_reject_invalid_before_runs(tmp_path, text):
    with pytest.raises(ValueError): read_combined_config(config(tmp_path,text))


def test_large_sweep_refusal_has_no_side_effects(tmp_path, monkeypatch):
    s=read_combined_config(config(tmp_path,'T=[1,2,3]\nm0=[0.1,0.2]\nwarning_threshold=2'))
    import combined_runs
    monkeypatch.setattr(combined_runs,'_run_program',lambda *a: pytest.fail('Started despite refusal'))
    before=set(tmp_path.iterdir())
    questions=[]
    def refuse(prompt):
        questions.append(prompt)
        return 'no'
    assert run_sweep(s,tmp_path/'combined.txt',confirm=refuse) is None
    assert len(questions)==1 and set(tmp_path.iterdir())==before


@pytest.mark.parametrize("asymmetry", [0., 1.])
def test_end_to_end_both_solvers_and_plots(tmp_path, asymmetry):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    path=config(tmp_path, 'T=[1.5,2.0]\nt_end=0.02\nn_spins=200\n'
                          f'record_interval=0.005\nfd_gaussian_order=8\nwarning_threshold=1\nasymmetry={asymmetry}')
    s=read_combined_config(path)
    prompts=[]
    def accept(prompt):
        prompts.append(prompt)
        return 'RUN'
    result=run_sweep(s,path,confirm=accept)
    assert len(prompts)==1
    m=load_manifest(result)
    assert m['state']=='complete' and len(m['cases'])==2
    for case in m['cases']:
        for method, entry in case['methods'].items():
            assert entry['status']=='complete'
            assert Path(entry['log']).exists()
            with np.load(entry['output'],allow_pickle=False) as d:
                params=json.loads(d['parameters'].item())
                assert all(params[k]==v for k,v in case['physical'].items())
                assert params['t_end']==.02
                assert d['trajectory'][-1,0]==pytest.approx(.02)
                assert len(d['energy'])==len(d['trajectory'])
    figs=plot_comparisons(m,show=False, show_equilibrium=False)
    assert len(figs)==2
    for fig, case in zip(figs,m['cases']):
        assert len(fig.axes)==2
        for panel in fig.axes:
            assert len(panel.lines)==2
        for idx, method in enumerate(['simulation','finite_difference']):
            with np.load(case['methods'][method]['output']) as d:
                np.testing.assert_array_equal(fig.axes[1].lines[idx].get_ydata(), d['energy'])
        plt.close(fig)


def test_failures_recorded_and_other_runs_continue(tmp_path, monkeypatch):
    import combined_runs
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    path=config(tmp_path,'T=[1.5,2.0]')
    s=read_combined_config(path)
    calls=[]
    def fail(script, parameter_file, log):
        calls.append(script.name)
        log.write_text('deliberate process-failure fixture')
        return 1
    monkeypatch.setattr(combined_runs,'_run_program',fail)
    manifest=load_manifest(run_sweep(s,path))
    assert calls==['simulate_spinglass.py','spinglass.py']*2
    assert manifest['state']=='complete_with_failures'
    assert all(e['status']=='failed' for c in manifest['cases'] for e in c['methods'].values())
    figs=plot_comparisons(manifest,show=False, show_equilibrium=False)
    for fig in figs:
        assert all('failed' in line.get_label() for ax in fig.axes for line in ax.lines)
        plt.close(fig)


def test_interrupt_preserves_manifest(tmp_path,monkeypatch):
    import combined_runs
    path=config(tmp_path)
    def interrupt(*args): raise KeyboardInterrupt
    monkeypatch.setattr(combined_runs,'_run_program',interrupt)
    with pytest.raises(KeyboardInterrupt): run_sweep(read_combined_config(path),path)
    manifest=load_manifest(next(tmp_path.glob('comparison_*/manifest.json')))
    assert manifest['state']=='interrupted'
    assert manifest['cases'][0]['methods']['simulation']['status']=='interrupted'


def test_shared_physical_input_names_and_temperature_alias(tmp_path):
    from spinglass import read_config as read_fd
    from simulate_spinglass import read_config as read_simulation
    physical=dict(temperature=.5,asymmetry=1.,j0=.4,j1=1.2,m0=.3)
    for temperature_name in ('T','temperature'):
        text=''.join(f'{temperature_name if k=="temperature" else k} = {v}\n' for k,v in physical.items())
        path=tmp_path/'shared.txt';path.write_text(text)
        fd=read_fd(path);sim=read_simulation(path);combined=read_combined_config(path)
        for key,value in physical.items():
            assert fd[key]==getattr(sim,key)==combined[key]==value
    path.write_text('T = .5\ntemperature = .5\n')
    for reader in (read_fd,read_simulation,read_combined_config):
        with pytest.raises(ValueError): reader(path)


def test_failed_partial_output_retained_and_plotted(tmp_path, monkeypatch):
    import combined_runs
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    path = config(tmp_path)
    def fail(script, parameter_file, log):
        log.write_text('deliberate failure')
        if script.name == 'spinglass.py':
            np.savez(parameter_file.parent/'finite_difference_test.npz',
                     trajectory=np.array([[.1, .3, .2, 1., 1e-6]]), energy=np.array([-.2]),
                     parameters=json.dumps(dict(run_status='failed', completed_time=.1)))
        return 1
    monkeypatch.setattr(combined_runs, '_run_program', fail)
    manifest = load_manifest(run_sweep(read_combined_config(path), path))
    entry = manifest['cases'][0]['methods']['finite_difference']
    assert manifest['state'] == 'complete_with_failures'
    assert entry['status'] == 'failed' and Path(entry['output']).exists()
    fig, = plot_comparisons(manifest, show=False, show_equilibrium=False)
    for ax in fig.axes:
        line = ax.lines[1]
        assert 'partial; failed at t=0.1' in line.get_label()
        np.testing.assert_array_equal(line.get_xdata(), [.1])
    plt.close(fig)


def test_comparison_figures_saved_without_display(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    manifest = dict(cases=[dict(index=3, physical=dict(temperature=.01),
                               methods=dict(finite_difference=dict(status='failed')))])
    destination = tmp_path/'figures'
    fig, = plot_comparisons(manifest, show=False, show_equilibrium=False, figures_dir=destination)
    image = destination/'case_0003.png'
    assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert image.stat().st_size > 1000
    assert 'failed' in fig.axes[0].lines[1].get_label()
    plt.close(fig)


@pytest.mark.parametrize('value', ['0', '-1', '1.5', '[2,3]', 'True'])
def test_invalid_ensemble_count(tmp_path, value):
    with pytest.raises(ValueError):
        read_combined_config(config(tmp_path, 'n_runs='+value))


def test_large_ensemble_requires_confirmation(tmp_path, monkeypatch):
    import combined_runs
    path = config(tmp_path, 'n_runs=11')
    monkeypatch.setattr(combined_runs, '_run_program', lambda *a: pytest.fail('Started without approval'))
    assert run_sweep(read_combined_config(path), path, confirm=lambda prompt: 'no') is None
    assert not list(tmp_path.glob('comparison_*'))


def test_combined_ensemble_end_to_end(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    path = config(tmp_path, 'n_runs=3\nn_spins=200\nt_end=.01\nrecord_interval=.005\nfd_gaussian_order=8')
    manifest = load_manifest(run_sweep(read_combined_config(path), path))
    case, = manifest['cases']
    assert manifest['state'] == 'complete'
    sim = case['methods']['simulation']
    with np.load(sim['output'], allow_pickle=False) as data:
        paths = data['trajectories'].copy()
        assert paths.shape[0] == 3
        metadata = json.loads(data['parameters'].item())
        assert metadata['n_runs'] == 3
        ensemble = json.loads(data['ensemble'].item())
        assert ensemble['coupling_seed'] == 0
        assert ensemble['run_seeds'] == [0, 1, 2]
    assert 'n_runs' not in Path(case['methods']['finite_difference']['parameter_file']).read_text()
    fig, = plot_comparisons(manifest, show=False, show_equilibrium=False, figures_dir=tmp_path/'figures')
    for panel, column in zip(fig.axes, [1, 2]):
        assert len(panel.lines) == 4
        for index in range(3):
            line = panel.lines[index]
            np.testing.assert_array_equal(line.get_ydata(), paths[index, :, column])
            assert line.get_color() == ('tab:blue' if column == 1 else 'tab:green')
            assert line.get_linewidth() == pytest.approx(2.0/3)
        assert panel.lines[-1].get_color() == 'tab:orange'
    assert (tmp_path/'figures/case_0001.png').is_file()
    plt.close(fig)


@pytest.mark.parametrize('n_runs', [3, 20, 30])
def test_notebooks_use_identical_ensemble_widths(tmp_path, monkeypatch, n_runs):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from combined_runs import simulation_linewidth
    paths = np.zeros((n_runs, 2, 5))
    paths[:, :, 0] = [0., .1]
    paths[:, :, 1] = np.arange(n_runs)[:, None] * .001
    paths[:, :, 2] = -.2
    archive = tmp_path/'simulation.npz'
    np.savez(archive, trajectory=paths[0], trajectories=paths, energy=paths[0,:,2],
             parameters=json.dumps(dict(n_runs=n_runs)))
    manifest = dict(cases=[dict(index=1, physical={}, methods=dict(
        simulation=dict(status='complete', output=str(archive))))])
    comparison, = plot_comparisons(manifest, show=False, show_equilibrium=False)
    cell = json.loads((Path(__file__).resolve().parents[1]/'inspect_trajectory.ipynb').read_text())['cells'][7]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    scope = dict(Path=Path, plt=plt, trajectory=paths[0], trajectories=paths,
                 energy=paths[0,:,2], filename=archive, parameters={}, is_simulation=True)
    exec(''.join(cell['source']), scope)
    inspection = scope['fig']
    for first, second in zip(comparison.axes, inspection.axes):
        for index in range(n_runs):
            assert first.lines[index].get_linewidth() == second.lines[index].get_linewidth()
            assert first.lines[index].get_linewidth() == simulation_linewidth(n_runs)
            np.testing.assert_array_equal(first.lines[index].get_ydata(), second.lines[index].get_ydata())
    assert simulation_linewidth(n_runs) == pytest.approx(max(.3, 2./n_runs))
    plt.close(comparison)
    plt.close(inspection)


def test_group_initial_magnetizations_with_partial_and_missing_results(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    cases = []
    for index, (temperature, m0) in enumerate([(1., -.3), (1., .4), (2., -.3)], 1):
        paths = np.zeros((2, 2, 5))
        paths[:, :, 0] = [0., .1]
        paths[:, :, 1] = m0
        paths[:, :, 2] = -temperature
        sim_path = tmp_path/f'sim{index}.npz'
        fd_path = tmp_path/f'fd{index}.npz'
        np.savez(sim_path, trajectories=paths, trajectory=paths[0],
                 energy=paths[0,:,2], parameters='{}')
        np.savez(fd_path, trajectory=paths[0], energy=paths[0,:,2],
                 parameters=json.dumps(dict(run_status='failed' if index == 2 else 'complete',
                                            completed_time=.1)))
        methods = dict(simulation=dict(status='complete', output=str(sim_path)),
                       finite_difference=dict(status='failed' if index == 2 else 'complete', output=str(fd_path)))
        if index == 3:
            methods['finite_difference'] = dict(status='failed')
        cases.append(dict(index=index, physical=dict(temperature=temperature, m0=m0,
                          j0=.5, j1=1., asymmetry=0.), methods=methods))
    figures = plot_comparisons(dict(cases=cases), show=False, show_equilibrium=False, figures_dir=tmp_path/'figures')
    assert len(figures) == 2
    for panel_index, panel in enumerate(figures[0].axes):
        assert len(panel.lines) == 6
        for offset, m0 in [(0, -.3), (3, .4)]:
            for line in panel.lines[offset:offset+2]:
                if panel_index == 0:
                    assert line.get_color() == 'tab:blue'
                else:
                    np.testing.assert_array_equal(line.get_color(), panel.lines[offset].get_color())
            fd = panel.lines[offset+2]
            assert fd.get_color() == 'tab:orange' and fd.get_linestyle() == '--'
            assert f'm0={m0:g}' in fd.get_label()
        assert 'partial' in panel.lines[-1].get_label()
    energy_panel = figures[0].axes[1]
    assert not np.array_equal(energy_panel.lines[0].get_color(), energy_panel.lines[3].get_color())
    assert energy_panel.get_legend().get_title().get_text() == 'Initial magnetisation m0'
    assert all(any(f'm0={value:g}' in text.get_text() for text in energy_panel.get_legend().get_texts())
               for value in [-.3, .4])
    assert (tmp_path/'figures/cases_0001_0002.png').exists()
    assert (tmp_path/'figures/case_0003.png').exists()
    assert 'failed' in figures[1].axes[0].lines[-1].get_label()
    for fig in figures:
        plt.close(fig)
