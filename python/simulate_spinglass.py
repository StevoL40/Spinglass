"""Random-sequential heat-bath simulation of the symmetric/asymmetric SK model."""
# Standard-library support for settings, command-line input, timing and files.
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import time
# NumPy supplies the arrays, random-number streams and histogram operations.
import numpy as np


# Reading guide
# -------------
# This is a finite-N microscopic simulation, independent of the RS diffusion
# solver in spinglass.py. Couplings are drawn once (quenched disorder); random
# single-site heat-bath attempts then evolve explicit +/-1 spins. One time unit
# means N ATTEMPTED updates, including rejected flips, not N accepted flips.
#
# Conventions: matrix[i,j] acts on spin i from spin j, fields = matrix @ spins.
# J0/N is the mean coupling and J1/sqrt(N) is the disorder scale. k=asymmetry
# mixes symmetric and antisymmetric disorder; it is not itself their reciprocal
# correlation. k=0 is the symmetric SK model; k=1 has zero reciprocal disorder
# correlation. There is no external field or self-coupling in this implementation.
#
# Flow through this file: validate inputs -> construct couplings and initial
# spins -> run one trajectory with cached local fields -> optionally repeat on
# the same disorder -> save arrays and provenance. Histograms are observations:
# their field limits and bin counts never constrain or discretize spin dynamics.
# The current directory determines relative output paths. From the project root:
#   python python/simulate_spinglass.py config/simulation_parameters.txt


@dataclass(frozen=True)
class Parameters:
    """One physical parameter set, recording schedule and ensemble configuration.

    Temperature and fields use coupling-energy units with Boltzmann k=1.
    Durations/recording intervals are attempted updates per spin and must resolve
    to integer attempt counts. cells/hmin/hmax affect histograms only. n_runs
    controls repeats sharing ONE coupling matrix, not independent disorder draws.
    """
    # Population and physical model settings.
    n_spins: int = 1000
    temperature: float = 1.5
    j0: float = 0.0
    j1: float = 1.0
    asymmetry: float = 0.0
    m0: float = 0.2
    # Duration and the two independent observation schedules.
    t_end: float = 10.0
    record_interval: float = 0.025
    distribution_interval: float = 2.0
    # Histogram geometry only: these settings do not limit actual local fields.
    hmin: float = -6.0
    hmax: float = 6.0
    cells: int = 120
    # Ensemble size, reproducible seed and base output name.
    n_runs: int = 1
    seed: int = 0
    output: str = 'Output/simulation.npz'

    def __post_init__(self):
        # Reject nonfinite numbers before checking counts and physical ranges.
        for name, value in asdict(self).items():
            if name != 'output' and not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        # Counts and seeds must be actual integers, not floating-point lookalikes.
        for name in ('n_spins', 'cells', 'seed', 'n_runs'):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f'{name} must be an integer')
        # Check physical, recording and output settings before allocating a matrix.
        if self.n_spins < 2 or self.cells < 1 or self.seed < 0 or self.n_runs < 1:
            raise ValueError('Require n_spins>=2, cells>=1, seed>=0 and n_runs>=1')
        if not 0 <= self.asymmetry <= 1:
            raise ValueError('Require 0<=asymmetry<=1 (thesis parameter k)')
        if self.temperature < 0 or self.j1 < 0 or abs(self.m0) > 1:
            raise ValueError('Require temperature>=0, j1>=0 and |m0|<=1')
        if self.t_end < 0 or self.record_interval <= 0 or self.distribution_interval <= 0:
            raise ValueError('Require t_end>=0 and positive recording intervals')
        if self.hmax <= self.hmin or not self.output:
            raise ValueError('Require hmax>hmin and a nonempty output path')
        # An attempted update is the smallest resolvable time unit.
        for name in ('t_end', 'record_interval', 'distribution_interval'):
            updates = getattr(self, name)*self.n_spins
            if not math.isclose(updates, round(updates), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f'{name} * n_spins must be an integer (attempted updates)')
        if round(self.record_interval*self.n_spins) < 1 or round(self.distribution_interval*self.n_spins) < 1:
            raise ValueError('Recording intervals must span at least one attempted update')


def read_config(path):
    """Parse scalar settings, canonicalize shared aliases and validate the result.

    Alias conversion precedes duplicate detection: T and temperature cannot both
    be supplied. Type conversion follows each default's type. Common output
    naming is resolved in parameter_files; no config-file expression is executed.
    Unlike the sweep reader, this standalone reader does not accept lists.
    """
    # Defaults provide both the allowed parameter names and conversion types.
    defaults = asdict(Parameters()) | dict(output_dir='Output', output_prefix='simulation')
    settings = {}
    # Read assignments with line numbers so errors identify the offending input.
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        # Remove surrounding whitespace and translate shared aliases first.
        name, separator, value = line.partition('=')
        name, value = name.strip(), value.strip()
        name = 'temperature' if name == 'T' else name
        name = 'cells' if name == 'simulation_cells' else name
        # Reject unknown names and duplicate aliases rather than overwriting them.
        if not separator or name not in defaults or name in settings:
            raise ValueError(f'{path}:{number}: unknown, duplicate or malformed parameter')
        # Convert using the default value's type; Parameters checks value ranges.
        try:
            settings[name] = type(defaults[name])(value)
        except ValueError:
            raise ValueError(f'{path}:{number}: invalid value for {name}') from None
    # Resolve output aliases, fill omitted defaults and validate the complete set.
    from parameter_files import resolve_output_settings
    return Parameters(**resolve_output_settings(settings, defaults['output']))


def make_couplings(p, rng):
    """Draw a dense quenched matrix with the normalization of thesis (1.18).

    For each unordered pair, independent standard normals S,A produce
    J_ij=J0/N + J1*(S+k*A)/sqrt(N*(1+k*k)) and the reverse has -k*A.
    Off-diagonal variance is J1**2/N for every k; reciprocal disorder correlation
    is (1-k*k)/(1+k*k). The zero diagonal excludes self-fields. Fortran order
    keeps columns contiguous for the O(N) cached-field update after each flip.
    Matrix storage is O(N**2); this is not a sparse approximation.
    """
    # Allocate the dense matrix with its diagonal initially fixed at zero.
    matrix = np.zeros((p.n_spins, p.n_spins), order='F')
    # Thesis equation (1.18), normalized to variance j1^2/N for every k.
    norm = math.hypot(1., p.asymmetry)
    symmetric = 1./norm
    antisymmetric = p.asymmetry/norm
    scale = p.j1/math.sqrt(p.n_spins)
    # Fill both directions of each pair together using the same two random draws.
    for i in range(p.n_spins-1):
        size = p.n_spins-i-1
        s, a = rng.standard_normal((2, size))
        matrix[i, i+1:] = p.j0/p.n_spins + scale*(symmetric*s + antisymmetric*a)
        matrix[i+1:, i] = p.j0/p.n_spins + scale*(symmetric*s - antisymmetric*a)
    return matrix


def initial_spins(p, rng):
    """Choose the nearest allowed population to m0, uniformly without replacement.

    Exactly `count` distinct sites start at -1. Finite N quantizes magnetization
    in steps of 2/N, so actual_initial_magnetization can differ from requested m0.
    """
    # Start with all +1, then choose exactly the required number of distinct -1 sites.
    spins = np.ones(p.n_spins, dtype=np.int8)
    count = math.floor(.5 + (1-p.m0)*p.n_spins/2)
    spins[rng.choice(p.n_spins, count, replace=False)] = -1
    return spins


def flip_probability(aligned_field, temperature):
    """Heat-bath probability for an attempted flip, with aligned_field=s_i*h_i.

    At zero temperature a misaligned spin always flips, an aligned spin never
    flips, and a zero-field spin flips with probability 1/2. The latter is the
    T->0 heat-bath convention, not an arbitrary tie-breaking sign rule.
    """
    if temperature == 0:
        return 1. if aligned_field < 0 else (0. if aligned_field > 0 else .5)
    return .5*(1-np.tanh(aligned_field/temperature))


def field_distribution(spins, fields, edges):
    """Observe rho[spin,bin] and excluded counts without changing the dynamics.

    Rows are +1 then -1. Divide counts by TOTAL N and bin width, not the size of
    each spin population: these are joint densities, not conditional histograms.
    Their integrated mass is less than one if fields lie outside the domain;
    report that loss instead of renormalizing. NumPy includes the last right
    edge, so only fields strictly above it count as upper overflow.
    """
    # Bin each spin population on the same edges and separately count exclusions.
    counts = np.stack([np.histogram(fields[spins == sign], edges)[0] for sign in (1, -1)])
    outside = np.array([[np.count_nonzero((spins == sign) & (fields < edges[0])),
                         np.count_nonzero((spins == sign) & (fields > edges[-1]))]
                        for sign in (1, -1)])
    return counts/(len(spins)*np.diff(edges)), outside


def run(p, progress_interval=10., *, _couplings=None):
    """Run one random-sequential trajectory and return arrays ready for saving.

    Each attempt chooses a site uniformly WITH replacement and tests its flip
    probability. Rejections still advance time by 1/N. _couplings is the
    ensemble's internal hook for sharing disorder; this routine never edits it.
    Results include t=0 and the final attempt, even when the regular sampling
    intervals do not divide the duration. The initial derivative is undefined
    (NaN); subsequent recorded derivatives are finite differences over samples.
    """
    # Separate streams keep the disorder independent of initialization and updates.
    streams = np.random.SeedSequence(p.seed).spawn(3)
    # Drawing more couplings must not consume initialization/update randomness.
    # Conversely changing output cadence consumes none of the three streams.
    coupling_rng, initial_rng, update_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams]
    started = last_report = time.monotonic()
    if _couplings is None:
        print('Generating couplings...', flush=True)
        matrix = make_couplings(p, coupling_rng)
    else:
        matrix = _couplings
    # Initialize this path's spins/fields, preserving the initial configuration.
    spins = initial_spins(p, initial_rng)
    initial = spins.copy()
    fields = matrix @ spins
    # Convert time settings to exact integer counts once. Scalar trajectory and
    # histogram samples have separate schedules, both independent of updates.
    edges = np.linspace(p.hmin, p.hmax, p.cells+1)
    total = round(p.t_end*p.n_spins)
    record_every = round(p.record_interval*p.n_spins)
    distribution_every = round(p.distribution_interval*p.n_spins)
    # Accumulate scalar samples and histogram snapshots in separate histories.
    records, densities, density_times, outside_history = [], [], [], []
    accepted = 0

    def record(attempt):
        """Measure m and -s.h/(2N) directly, independently of histogram bounds."""
        t = attempt/p.n_spins
        m = float(spins.mean())
        energy = -.5*np.dot(spins.astype(float), fields)/p.n_spins
        # For symmetric J this is Hamiltonian energy per spin. For directed J,
        # s.T@J@s contains only J's symmetric part, while h_i and transition
        # rates depend on the full J: this is not generally a Lyapunov energy.
        # Estimate rates between samples; the initial sample has no predecessor.
        derivatives = [np.nan, np.nan] if not records else [(m-records[-1][1])/(t-records[-1][0]),
                                                            (energy-records[-1][2])/(t-records[-1][0])]
        records.append([t, m, energy, *derivatives])

    def distribution(attempt):
        """Snapshot the joint histogram and separately preserve excluded mass."""
        density, outside = field_distribution(spins, fields, edges)
        densities.append(density)
        outside_history.append(outside)
        density_times.append(attempt/p.n_spins)

    # Record the initial state even for a zero-duration run.
    record(0)
    distribution(0)
    print(f'Starting simulation: t=0/{p.t_end:g}, attempts=0', flush=True)
    # Attempt exactly the requested number of updates, counting rejected flips too.
    for attempt in range(1, total+1):
        site = int(update_rng.integers(p.n_spins))
        if update_rng.random() < flip_probability(float(spins[site])*fields[site], p.temperature):
            delta = -2*int(spins[site])
            spins[site] = -spins[site]
            # Only source spin `site` changed, so all affected fields receive
            # delta*J[:,site]. Using the ROW would be wrong for asymmetric J.
            # The site's own field is unchanged because the diagonal is zero.
            fields += delta*matrix[:, site]
            accepted += 1
        # Refresh independently of recording cadence to limit accumulated roundoff.
        if attempt % p.n_spins == 0 or attempt == total:
            fields = matrix @ spins
        # Observe the current state on each schedule, always including the endpoint.
        if attempt % record_every == 0 or attempt == total:
            record(attempt)
        if attempt % distribution_every == 0 or attempt == total:
            distribution(attempt)
        # Throttle progress messages by wall-clock time without changing the path.
        now = time.monotonic()
        if now-last_report >= progress_interval:
            print(f'Running: t={attempt/p.n_spins:g}/{p.t_end:g}, '
                  f'attempts={attempt}/{total}, elapsed={now-started:.1f}s', flush=True)
            last_report = now
    # Finish reporting and package the observed arrays for the caller/archive.
    print(f'Simulation complete: attempts={total}, accepted={accepted}, elapsed={time.monotonic()-started:.1f}s', flush=True)
    trajectory = np.asarray(records)
    # trajectory rows: [time, m, energy, dm/dt, dE/dt]. Histogram histories use
    # their own rho_times; do not assume they align with every trajectory row.
    # There is no replica overlap q measurement in this microscopic simulator.
    return dict(trajectory=trajectory, trajectory_columns=np.array(['time', 'm', 'energy', 'dm_dt', 'dE_dt']),
                energy=trajectory[:, 2].copy(), h=(edges[:-1]+edges[1:])/2, field_edges=edges,
                rho=densities[-1], rho_history=np.asarray(densities), rho_times=np.asarray(density_times),
                outside_counts=np.asarray(outside_history), spins=spins, fields=fields, initial_spins=initial,
                attempted_updates=np.array(total), accepted_updates=np.array(accepted),
                run_type=np.array('spin_simulation'),
                simulation=np.array(json.dumps(dict(method='random_sequential_heat_bath_v1',
                    rng='PCG64', streams=['couplings', 'initialization', 'updates'],
                    zero_temperature_zero_field_flip_probability=.5,
                    actual_initial_magnetization=float(initial.mean()),
                    field_refresh_attempts=p.n_spins))))


def run_ensemble(p, progress_interval=10.):
    """Independent initial/update streams conditioned on one quenched disorder sample.

    Runs execute sequentially and reuse one immutable dense coupling matrix.
    Legacy single-run fields remain aliases of run zero.

    The base seed constructs the shared matrix once; run index i uses seed+i
    for its initialization/update streams. These are independent stochastic
    paths conditional on one disorder realization, not a disorder ensemble.
    n_runs=1 delegates directly to run(), preserving the single-run format.
    """
    # Build and protect shared disorder only when more than one path is requested.
    matrix = None
    if p.n_runs > 1:
        print('Generating shared couplings...', flush=True)
        stream = np.random.SeedSequence(p.seed).spawn(3)[0]
        matrix = make_couplings(p, np.random.Generator(np.random.PCG64(stream)))
        matrix.flags.writeable = False
    # Run each path sequentially with its own seed and the same optional matrix.
    results = []
    for index in range(p.n_runs):
        seed = p.seed + index
        print(f'Run {index+1}/{p.n_runs}: coupling seed={p.seed}, '
              f'initialization/update seed={seed}', flush=True)
        results.append(run(replace(p, seed=seed, n_runs=1), progress_interval,
                           _couplings=matrix))
    first = results[0].copy()
    # Retain all old single-run keys for readers expecting one path. Additional
    # arrays stack runs on a new leading axis; ordinary `trajectory` remains
    # run zero rather than an ensemble mean. Couplings are never stacked/copied.
    # Keep single-run archives unchanged; ensemble arrays are additional fields.
    if p.n_runs > 1:
        for key in ('trajectory', 'energy', 'rho', 'rho_history', 'outside_counts',
                    'spins', 'fields', 'initial_spins', 'attempted_updates', 'accepted_updates'):
            name = 'trajectories' if key == 'trajectory' else 'run_' + key
            first[name] = np.stack([result[key] for result in results])
        # Preserve per-path seed/method records and describe the ensemble itself.
        first['run_seeds'] = np.asarray([str(p.seed+i) for i in range(p.n_runs)])
        first['run_simulation'] = np.asarray([result['simulation'].item() for result in results])
        first['ensemble'] = np.array(json.dumps(dict(
            method='fixed_disorder_independent_initialization_and_updates_v1',
            n_runs=p.n_runs, coupling_seed=p.seed,
            run_seeds=[p.seed+i for i in range(p.n_runs)], legacy_fields_run_index=0)))
    return first


def reserve_output(output, started):
    """Reserve a timestamped NPZ path by exclusive creation; never overwrite.

    Add a numeric suffix on collisions. The parent directory must already exist;
    output naming does not create folders or resolve paths relative to a config.
    """
    # Construct a timestamped filename, accepting a base name with or without .npz.
    path = Path(output)
    stem = path.name[:-4] if path.name.endswith('.npz') else path.name
    base = f'{stem}_{started:%Y-%m-%d_%H%M%S_%f}'
    counter = 0
    # Claim the path exclusively; on a collision try the next numbered suffix.
    while True:
        candidate = path.with_name(f'{base}{"_"+str(counter) if counter else ""}.npz')
        try:
            return candidate, candidate.open('xb')
        except FileExistsError:
            counter += 1


def main(argv=None):
    """Load one config, simulate its ensemble and write a compressed NPZ archive.

    Record every effective setting and NumPy version beside numeric arrays.
    A single start timestamp names the run; reserve_output handles collisions.
    No pickle is needed: structured metadata is stored as JSON scalar strings.
    """
    # Declare the required config-file argument and read the command line.
    # argparse also handles --help and reports invalid command syntax.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parameter_file', help='text file containing name = value settings')
    args = parser.parse_args(argv)
    # Use one local start timestamp for filenames and saved provenance.
    started = datetime.now().astimezone()
    try:
        # Load/validate the settings and require an existing output directory.
        p = read_config(args.parameter_file)
        if not Path(p.output).parent.is_dir():
            raise ValueError('Output directory must already exist')
    except (ValueError, OSError) as error:
        # Turn input/file problems into a concise CLI error before simulation.
        parser.error(str(error))
    # Perform the requested paths before reserving their final output file.
    result = run_ensemble(p)
    # Record the actual destination and all effective settings alongside results.
    path, handle = reserve_output(p.output, started)
    metadata = asdict(p) | dict(config=str(args.parameter_file), output_file=str(path),
                               run_started=started.isoformat(), numpy_version=np.__version__)
    # Write a compressed archive and close the reserved file on leaving the block.
    with handle:
        np.savez_compressed(handle, **result, parameters=json.dumps(metadata))
    # Show the exact timestamped path so it can be selected in the notebooks.
    print(f'Saved {path}', flush=True)


# Execute the CLI on direct invocation, but not when importing simulation helpers.
if __name__ == '__main__':
    main()
