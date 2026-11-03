"""Parameter sweeps and comparison plots used by compare_trajectories.ipynb.

The existing command-line programs remain the authority for running each method.
"""
import ast
from dataclasses import asdict
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from spinglass import Parameters as FDParameters
from simulate_spinglass import Parameters as SimulationParameters

PHYSICAL = ('temperature', 'asymmetry', 'j0', 'j1', 'm0')
DEFAULTS = dict(temperature=1.5, asymmetry=0., j0=0., j1=1., m0=.3,
                t_end=1., hmin=-6., hmax=6., n_spins=1000, n_runs=1, seed=0,
                record_interval=.025, distribution_interval=2., simulation_cells=120,
                fd_cells=24, fd_gaussian_order=24, fd_dt_max=.002,
                fd_saddle_tol=2e-5, fd_saddle_maxiter=120,
                warning_threshold=10, output_dir='Output', output_prefix='comparison')


def read_combined_config(path):
    """Parse scalars/lists safely, with T as an alias for temperature."""
    settings = DEFAULTS.copy()
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name, sep, value = line.partition('=')
        name, value = name.strip(), value.strip()
        name = 'temperature' if name == 'T' else name
        location = f'{path}:{number}'
        if not sep or name not in settings or name in seen:
            raise ValueError(f'{location}: unknown, duplicate or malformed parameter {name!r}')
        seen.add(name)
        if name in ('output_dir', 'output_prefix'):
            if not value:
                raise ValueError(f'{location}: empty {name}')
            settings[name] = value
            continue
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f'{location}: expected a number or a physical-parameter list') from None
        values = parsed if isinstance(parsed, list) else [parsed]
        if isinstance(parsed, list) and name not in PHYSICAL:
            raise ValueError(f'{location}: lists are allowed only for {", ".join(PHYSICAL)}')
        if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f'{location}: require nonempty finite numerical values')
        if isinstance(settings[name], int) and type(parsed) is not int:
            raise ValueError(f'{location}: {name} must be an integer')
        if len(values) != len(set(values)):
            raise ValueError(f'{location}: duplicate list values would repeat a run')
        settings[name] = parsed
    validate_settings(settings)
    return settings


def axes(settings):
    return {name: settings[name] if isinstance(settings[name], list) else [settings[name]]
            for name in PHYSICAL}


def combination_count(settings):
    return math.prod(len(values) for values in axes(settings).values())


def physical_sets(settings):
    for values in itertools.product(*axes(settings).values()):
        yield dict(zip(PHYSICAL, values))


def method_settings(settings, physical, method):
    shared = physical | {key: settings[key] for key in ('t_end', 'hmin', 'hmax')}
    if method == 'simulation':
        return shared | {key: settings[key] for key in ('n_spins', 'n_runs', 'seed', 'record_interval', 'distribution_interval')} | dict(cells=settings['simulation_cells'])
    if method != 'finite_difference':
        raise ValueError(f'Unknown method: {method}')
    return shared | {key[3:]: value for key, value in settings.items() if key.startswith('fd_')}


def validate_settings(settings):
    if settings['warning_threshold'] < 1:
        raise ValueError('warning_threshold must be at least 1')
    prefix = settings['output_prefix']
    if not prefix or prefix in ('.', '..') or Path(prefix).name != prefix or '\\' in prefix:
        raise ValueError('output_prefix must be a simple filename prefix')
    if not Path(settings['output_dir']).is_dir():
        raise ValueError('output_dir must already exist (relative to the notebook working directory)')
    # Constraints in the current models are separable in the five physical axes.
    # Validate each value without materializing a potentially enormous product.
    base = {name: values[0] for name, values in axes(settings).items()}
    for name, values in axes(settings).items():
        for value in values:
            physical = base | {name: value}
            sim = method_settings(settings, physical, 'simulation')
            SimulationParameters(**sim)
            fd = method_settings(settings, physical, 'finite_difference')
            FDParameters(**{k: v for k, v in fd.items() if k in asdict(FDParameters())})
    # The FD program accepts this range, but its initializer needs both populations.
    # Endpoint m0 values are retained and any actual solver failure is reported.


def print_preview(settings):
    count = combination_count(settings)
    print('Physical parameter axes:')
    for key, values in axes(settings).items():
        print(f'  {key} = {values}')
    print(f'{count} parameter set(s); {count*settings["n_runs"]} simulation trajectories '
          f'and {count} finite-difference solutions, executed sequentially.')
    print(f'Duration per run: {settings["t_end"]:g} attempted updates per spin / FD time units.')
    print(f'Simulation: N={settings["n_spins"]}, seed={settings["seed"]}; '
          f'coupling matrix alone is approximately {8*settings["n_spins"]**2/1e6:.1f} MB.')
    print(f'FD: cells={settings["fd_cells"]}, Gaussian order={settings["fd_gaussian_order"]}, '
          f'budget={settings["fd_saddle_maxiter"]} objective evaluations per saddle.')
    print(f'Confirmation required when parameter sets × n_runs exceeds {settings["warning_threshold"]}.')
    print('Runtime also depends strongly on duration, N and numerical resolution; no reliable ETA is assumed.')


def _run_program(script, parameter_file, log_file):
    """Stream progress into the notebook and retain an independent per-method log."""
    with log_file.open('w') as log:
        with subprocess.Popen([sys.executable, str(script), str(parameter_file)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                return process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def run_sweep(settings, config_path, confirm=None):
    """Run both programs for each Cartesian set; fail closed on refused confirmation.

    Return None without writing anything when a large sweep is declined.
    """
    validate_settings(settings)
    count = combination_count(settings)
    if count * settings['n_runs'] > settings['warning_threshold']:
        print(f'WARNING: {count} combinations require {count*settings["n_runs"]} simulation '
              f'trajectories and {count} finite-difference solutions and may take a long time.')
        response = (confirm or input)('Type RUN to proceed; any other response cancels: ')
        if response.strip() != 'RUN':
            print('Sweep cancelled. No runs started or output directories created.')
            return None
    started = datetime.now().astimezone()
    parent = Path(settings['output_dir']).resolve()
    stem = f'{settings["output_prefix"]}_{started:%Y-%m-%d_%H%M%S_%f}'
    directory = parent/stem
    counter = 0
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            counter += 1
            directory = parent/f'{stem}_{counter}'
    manifest_path = directory/'manifest.json'
    manifest = dict(config=str(Path(config_path).resolve()), settings=settings,
                    run_started=started.isoformat(), directory=str(directory),
                    state='running', combination_count=count, cases=[])

    def save_manifest():
        temporary = directory/'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(manifest_path)

    save_manifest()
    script_dir = Path(__file__).resolve().parent
    try:
        for index, physical in enumerate(physical_sets(settings), 1):
            print(f'\nSet {index}/{count}: {physical}', flush=True)
            case_dir = directory/f'case_{index:04d}'
            case_dir.mkdir()
            case = dict(index=index, physical=physical, methods={})
            manifest['cases'].append(case)
            save_manifest()
            for method, script in [('simulation', 'simulate_spinglass.py'),
                                   ('finite_difference', 'spinglass.py')]:
                values = method_settings(settings, physical, method) | dict(output=str(case_dir/f'{method}.npz'))
                parameter_file = case_dir/f'{method}_parameters.txt'
                parameter_file.write_text(''.join(f'{key} = {value}\n' for key, value in values.items()))
                entry = dict(status='running', parameter_file=str(parameter_file), log=str(case_dir/f'{method}.log'))
                case['methods'][method] = entry
                save_manifest()
                print(f'Running {method}...', flush=True)
                code = _run_program(script_dir/script, parameter_file, Path(entry['log']))
                outputs = sorted(case_dir.glob(f'{method}_*.npz'))
                entry['returncode'] = code
                if code == 0 and len(outputs) == 1:
                    entry.update(status='complete', output=str(outputs[0]))
                else:
                    entry.update(status='failed', error=f'Exit code {code}; found {len(outputs)} output archives. See log.')
                    if len(outputs) == 1:
                        entry['output'] = str(outputs[0])
                    print(f'{method} FAILED for set {index}. Other methods/sets will continue; see {entry["log"]}.')
                save_manifest()
        manifest['state'] = 'complete_with_failures' if any(
            e['status'] == 'failed' for c in manifest['cases'] for e in c['methods'].values()) else 'complete'
    except BaseException:
        manifest['state'] = 'interrupted'
        for case in manifest['cases']:
            for entry in case['methods'].values():
                if entry['status'] == 'running':
                    entry['status'] = 'interrupted'
        raise
    finally:
        save_manifest()
        print(f'Manifest saved: {manifest_path}')
    return manifest_path


def load_manifest(path):
    return json.loads(Path(path).read_text())


def simulation_linewidth(n_runs):
    """Shared ensemble path width in points for both trajectory notebooks."""
    return max(0.3, 2.0 / n_runs)


def plot_comparisons(manifest, show=True, figures_dir=None, show_equilibrium=True):
    """One two-panel m/E figure per physical group, overlaying initial magnetizations."""
    import matplotlib.pyplot as plt
    figures = []
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for case in manifest['cases']:
        key = tuple(sorted((name, value) for name, value in case['physical'].items()
                           if name != 'm0'))
        groups.setdefault(key, []).append(case)
    for physical, cases in groups.items():
        magnetizations = sorted({case['physical'].get('m0') for case in cases},
                                key=lambda value: (value is not None, value))
        energy_colors = ({magnetizations[0]: 'tab:green'} if len(magnetizations) == 1 else
                         dict(zip(magnetizations, plt.get_cmap('Greens')(
                             np.linspace(.5, .9, len(magnetizations))))))
        fig, panels = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        for case in cases:
            energy_color = energy_colors[case['physical'].get('m0')]
            for method, label, style in [('simulation', 'Spin simulation', '-'),
                                         ('finite_difference', 'Finite difference', '--')]:
                if 'm0' in case['physical']:
                    label += f", m0={case['physical']['m0']:g}"
                entry = case['methods'].get(method, {})
                if entry.get('status') != 'complete' and not entry.get('output'):
                    for panel in panels:
                        panel.plot([], [], style, label=f'{label}: {entry.get("status", "not run")} (see log)')
                    continue
                with np.load(entry['output'], allow_pickle=False) as data:
                    metadata = json.loads(data['parameters'].item())
                    if entry.get('status') != 'complete' or metadata.get('run_status') == 'failed':
                        label += f" (partial; failed at t={metadata.get('completed_time', 0):.6g})"
                    trajectory = data['trajectory']
                    energy = data['energy']
                    if method == 'simulation' and 'trajectories' in data.files:
                        paths = data['trajectories']
                        line_width = simulation_linewidth(len(paths))
                        print(f'Simulation plot: {len(paths)} runs; line width={line_width:g} pt; {entry["output"]}')
                        for index, path in enumerate(paths):
                            path_label = f'{label} ({len(paths)} runs)' if index == 0 else None
                            panels[0].plot(path[:, 0], path[:, 1], color='tab:blue',
                                           linewidth=line_width, label=path_label)
                            panels[1].plot(path[:, 0], path[:, 2], color=energy_color,
                                           linewidth=line_width, label=path_label)
                    elif trajectory.size:
                        panels[0].plot(trajectory[:, 0], trajectory[:, 1], style, color='tab:orange' if method == 'finite_difference' else 'tab:blue', label=label)
                        panels[1].plot(trajectory[:, 0], energy, style, color='tab:orange' if method == 'finite_difference' else energy_color, label=label)
                    else:
                        # FD t_end=0 has no trajectory row; display the initial density observables.
                        rho, h = data['rho'], data['h']
                        dh = h[1]-h[0]
                        panels[0].plot([0], [dh*(rho[0]-rho[1]).sum()], 'o', label=label+' (initial)')
                        panels[1].plot([0], [-.5*dh*np.dot(h,rho[0]-rho[1])], 'o', label=label+' (initial)')
        if show_equilibrium:
            from rs_equilibrium import add_equilibrium_references
            add_equilibrium_references(panels[0], panels[1], [case['physical'] for case in cases])
        panels[0].set_ylabel('Magnetization m')
        panels[1].set_ylabel('Energy per spin E')
        panels[1].set_xlabel('Time (attempted updates per spin for simulation)')
        for panel in panels:
            panel.grid(True, alpha=.3)
            legend = panel.legend(title='Initial magnetisation m0' if panel is panels[1] else None)
            # Readable colour keys without changing the ensemble path widths.
            for handle in legend.get_lines():
                handle.set_linewidth(max(1.5, handle.get_linewidth()))
        title = ', '.join(f'{key}={value:g}' for key, value in physical)
        indices = ', '.join(str(case['index']) for case in cases)
        fig.suptitle(f'Set(s) {indices}: {title}')
        figures.append(fig)
        if figures_dir is not None:
            stem = ('case_' if len(cases) == 1 else 'cases_') + '_'.join(
                f"{case['index']:04d}" for case in cases)
            figure_path = figures_dir/f'{stem}.png'
            fig.savefig(figure_path, dpi=200, bbox_inches='tight')
            print(f'Saved figure: {figure_path}')
        if show:
            plt.show()
    return figures
