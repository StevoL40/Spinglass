# Numerical investigations, bug fixes and validation

This is the consolidated record of investigations previously kept in
`SADDLE_INVESTIGATION.md`, `SADDLE_VALIDATION.md` and `FLUX_VALIDATION.md`.
It covers the original C comparison, replacement saddle solver, subsequent
robustness and efficiency work, and the equilibrium-preserving transport fix.
The former filenames are retained only as pointers to this document.

## Reading this history

**Historical trajectories are not current reference predictions.** Sections 1-6
record the development before the 23 October 2026 spatial-flux correction.
Their numerical tables, failures and rejected experiments are preserved as
historical evidence. Statements that a change left transport unchanged apply
only to that development stage. Section 7 records the later flux correction;
the current solver combines it with the safeguarded saddle solver.

An accepted discretized saddle is not proof of continuum convergence or replica
stability. Completion, equation accuracy, numerical refinement and agreement
with physical references are separate checks. Test counts and manual run results
below describe the validation when recorded; consolidation did not rerun them.

| Investigation | Change or decision | Evidence and limitations |
| --- | --- | --- |
| Negative-r-squared proposals | Replace the earlier fixed-point iteration rather than restore C fallbacks | Historical source audit and isolated fallback experiments; see section 1 |
| Coupled saddle solve | PSD Gaussian covariance, bounded least squares and independent checks of every equation | Residual definitions and refinement tables; see section 2 |
| False convergence | Optimize original candidate-measure scalar residuals | Saved late-time failure and quadrature comparisons; see section 3 |
| Low-temperature underflow | Persistent logweights, density-based initialization and checked early stopping | Underflow checkpoint and higher-order forward runs; see section 4 |
| Fully asymmetric endpoint | Remove the coordinate singularity at k=1 | Endpoint and interior regression checks; see section 5 |
| Tail velocities, difficult fits and cost | Consistent conditional transport, bounded SVD recovery, factored measure and safeguarded initialization | Checkpoints, rejected alternatives and unresolved forward failures; see section 6 |
| Equilibrium energy bias | Replace artificial-diffusion-prone upwind flux by exponential fitting | Analytic equilibrium, full-flow and refinement checks; see section 7 |

### Contents

1. [Initial fixed-point investigation](#1-initial-fixed-point-investigation)
2. [Coupled saddle solver and initial validation](#2-coupled-saddle-solver-and-initial-validation)
3. [Long-run objective correction](#3-long-run-objective-correction-19-october-2026)
4. [Low-temperature numerical-state correction](#4-low-temperature-numerical-state-correction-19-october-2026)
5. [Asymmetry and the fully asymmetric endpoint](#5-asymmetry-and-the-fully-asymmetric-endpoint)
6. [Strong-mean-coupling stability and efficiency](#6-strong-mean-coupling-stability-and-efficiency)
7. [Equilibrium-preserving field transport](#7-equilibrium-preserving-field-transport-23-october-2026)
8. [Current interpretation and remaining limits](#8-current-interpretation-and-remaining-limits)

## 1. Initial fixed-point investigation

This section concerns the superseded fixed-point Python solver. Its radicand
exceptions, damping and stopping tests are historical behaviour, not descriptions
of the present coupled solver. The historical negative-r-squared failure did
not establish that an admissible saddle was absent.

### Source and scope

Source: [SL_DPhil_Thesis_1995.pdf](docs/SL_DPhil_Thesis_1995.pdf). Page references below give the printed thesis
page followed by the one-based PDF page number. This is a targeted comparison of
the replica-symmetric measure, moment contractions, multiplier update, and
iteration failure handling. It is not a validation of the complete reconstruction
or a reproduction of the thesis figures.

The relevant replica-symmetric equations are in Chapter 6, section 6.4.1,
equation (6.15), pp. 93-94 (PDF pp. 105-106). Chapter 7 concerns replica symmetry
breaking; it is not the primary source of these implemented RS equations.

### Confirmed comparisons

- Appendix D, `calcstuff`, p. 149 (PDF p. 161), lines approximately 340-350:
  `thetaoverr = OK*Theta_1/r` when `r != 0`, otherwise zero. For positive
  `rtl = lambda_1*lambda_1 - thetaoverr*thetaoverr`, the code uses `sqrt(rtl)`;
  otherwise it uses `lambda_1`. Python instead stops for a materially negative
  radicand. The positive-radicand exponent signs and normalization structure
  match the Python measure, subject to the different quadrature and field grid.
- Appendix D, `calcmmur`, pp. 153-154 (PDF pp. 165-166), lines approximately
  630-645: the expressions for `mu`, `Theta_0`, `Theta_1`, and `rsquared` match
  the Python expressions. No transcription sign error was found in that update.
  The C code saves `oldr` and retains it if `rsquared < 0`. Python instead raises
  an exception for values below `-1e-7` and clips smaller negative values to zero.
- Appendix D, `calcrho` and `calct`, pp. 155-158 (PDF pp. 167-170): pair-density
  contractions for spin and field moments have the structure of Python's `sr`
  and `hr`; the `rterm` and `tterm` contractions match. Python factors these
  contractions rather than allocating the full pair-density matrices.
- Appendix D, `saddle`, p. 153 (PDF p. 165), lines approximately 595-620:
  the historical update order and additive chi update differ from Python's
  damped multiplicative fixed-point iteration. Restoring two fallbacks alone
  does not reproduce the historical algorithm in full.

### Reproduction with the historical unmodified solver

Physical settings: `j0=0`, `j1=1`, `temperature=1.5`, `asymmetry=0`, `m0=0.2`.
Numerical settings: `hmin=-6`, `hmax=6`, `cells=24`, `gaussian_order=4`, other
solver defaults unless specified below.

| dt_max | t_end | Observed result |
| --- | --- | --- |
| 0.002 | 0.001 | One step; initial saddle accepted without an update |
| 0.002 | 0.0011 | One step; completes |
| 0.002 | 0.002 | One step; completes |
| 0.002 | 0.01 | Second saddle fails at t=0.002, proposed rsquared=-0.0001131451412 |
| 0.001 | 0.01 | Second saddle fails at t=0.001, proposed rsquared=-0.00005667713653 |
| 0.0001 | 0.01 | Fifth saddle fails at t=0.0004, proposed rsquared=-0.00002269210620 |

At the failing second solve with `dt_max=0.002`, the density residual is
approximately `0.000123000588`, above `saddle_tol=0.00002`. The update starts from
the previous saddle measure while using the evolved density's moments. The
second field moment has changed to approximately `1.00136271445`; the old measure
still gives `rterm` approximately `0.0399999960054`. The three terms in the proposed
rsquared are approximately `0.04`, `-0.0401135638649`, and `0.000000418723719`.
Thus the failing value is an intermediate fixed-point proposal, not a proof that
no admissible saddle exists. The absence of an earlier progress message reflects
the short elapsed wall-clock time, not failure at the first timestep.

### Isolated experiments, not applied to spinglass.py

Replacing only the negative-rsquared exception with the historical old-r fallback
allows `t_end=0.01` to complete in five steps. Extending to `0.1` or `1` then hits
the Gaussian-radicand exception at `t=0.01`.

Replacing both checks with the historical fallback rules produces:

| t_end | Steps | Final magnetization | Last recorded q | Last density residual |
| --- | --- | --- | --- | --- |
| 0.01 | 5 | 0.198012435 | 0.0393923900 | 0.00000296634598 |
| 0.1 | 50 | 0.181161179 | 0.0331303330 | 0.00000414952622 |
| 1.0 | 500 | 0.0822623666 | 0.0118159603 | 0.0000174775640 |

These experiments retain Python's quadrature, flux method, damping, and stopping
test. They are not validated physical predictions. In particular, at the last
accepted saddle of the `0.01` run, recomputing the multiplier equation yields
rsquared approximately `-0.00000251788311` while the stored r is zero. Passing the
density and q tests therefore does not establish satisfaction of every saddle
equation. Longer experiments also show a nonzero multiplier-equation mismatch.

### Decision and outcome


Two alternatives were considered at the time:

1. Restore the two documented Appendix D fallback rules in the existing Python
   iteration, expose their use in diagnostics, and describe this as historical
   numerical behaviour with explicit validation limitations.
2. Develop and validate a solver with residual checks for all saddle equations
   and admissibility conditions. This requires broader numerical-method work;
   the thesis code's fallbacks alone do not establish an acceptable solution.

The second alternative was selected and implemented in section 2. The first
alternative remained an isolated experiment. The investigation required retaining
file input, timestamped output, metadata, plotting compatibility and conservation
checks, without silently clipping negative proposals or treating shorter successful
runs as proof of a fix.

## 2. Coupled saddle solver and initial validation

The replacement checks every saddle equation instead of retaining the C code's
intermediate-iterate fallbacks. The representation and acceptance rules below
remain relevant; the early trajectory tables were obtained before later
initialization and spatial-flux improvements.

### Equations and representation

The source is section 6.4.1, equation (6.15), printed pp. 93-94 (PDF pp. 105-106)
of [SL_DPhil_Thesis_1995.pdf](docs/SL_DPhil_Thesis_1995.pdf); Appendix D's corresponding routines are on printed
pp. 149 and 153-158. The notation maps as follows:

| Thesis | Python |
| --- | --- |
| J tilde | `j1` |
| k | `asymmetry` |
| R0, R1 | `theta0`, `theta1` |
| Q0 | `r**2` |
| lambda1 squared | `q / (j1**2 * (1-q)**2)` |

The random field and spin terms in the RS exponent have covariance

```text
[[lambda1**2,       (1-k**2)*R1],
 [(1-k**2)*R1,      Q0         ]].
```

Represent them by `field_noise = lambda1*z1` and
`spin_noise = b*z1 + sqrt(v)*z2`, with independent standard normals and `v >= 0`.
Then `(1-k**2)*R1 = lambda1*b` and `Q0 = b*b + v`. This is the same real Gaussian
integral with a different factorization; it is well-defined at the initial
`Q0 = R1 = 0` state and never takes the square root of a negative proposal.
No old-r or lambda1 fallback is used.

The density exponent absorbs `mu + (1-k**2)*R0*(h-j0*m)` into a separate effective
logweight for each spin/field cell. One reference logweight is fixed to remove
the irrelevant common scale. `mu` and `R0` are eliminated algebraically using
the target density's moments. The remaining logweights, `q`, `b`, and `v` are
solved simultaneously by SciPy's bounded least-squares method. The objective
includes every density cell and the original magnetization, mu, q, R0, R1 and Q0
equations evaluated with candidate-measure moments. The exact roots are unchanged
by these redundant moment constraints, but the finite-error fitting criterion now
agrees with the independent acceptance check. Only cells with positive target
density are active.
The Gaussian expectations and spatial flux method retain the existing quadrature
order and cell-centred grid. The initialization of `rterm` now includes `j1**2`,
consistent with the analytic initial field-pair moment for non-unit coupling.

Analytic logweight derivatives avoid reevaluating the Gaussian integral separately
for every cell; the three covariance variables use finite differences.
The optimizer is bounded by `0 <= q < 1` and `v >= 0`. Its status flag is not the
acceptance criterion. `saddle_maxiter` is now the objective-evaluation budget;
`saddle_damping` remains accepted and recorded but is inactive.

### Independent acceptance check

`SpinGlass.saddle_residuals` recomputes equation (6.15) from the candidate measure's
own moments in a separate implementation from the optimization evaluation. It checks:

- Every spin/field density cell, using the maximum error multiplied by `j1`.
- Magnetization, `mu`, and `q`.
- `R0` and `R1`, with equation errors multiplied by `j1`.
- `Q0`, using its original equation, without a square root.
- Conditional normalization at every Gaussian node.
- Covariance admissibility, using the negative part of the dimensionless
  covariance determinant. Trial covariances are already PSD by construction.

All residuals must be finite with absolute value at most `saddle_tol`. Density,
normalization and covariance entries are nonnegative norms; the other entries
are signed equation errors. A failed solve raises an exception with equation
details and does not commit trial multipliers or logweights.

New output files save all these residuals, their names, the actual pre-update
saddle times, and solver identification. The trajectory's existing fifth column
now contains the maximum absolute equation residual instead of density RMS error.
Its q and residual still refer to the saddle before the density update; the
time, magnetization and mass columns refer to the updated density. The final
saved density does not trigger an extra saddle solve. For the refinement table
below, an extra saddle was explicitly solved on each final density to compare
q at the same time, rather than at different pre-update times.

### Checks performed

The automated checks are in `tests/test_spinglass.py`; run `python -m pytest -q`.
They include:

- Five-step regression of the reported `t_end=0.01` failure, checking all accepted
  equations, probability mass, nonnegative density, and saved diagnostics.
- The analytic high-temperature, zero-magnetization equilibrium of (6.19)-(6.20):
  `q=Q0=R1=mu=0`, `R0=beta/2`, with the two Gaussian field distributions centred
  at `+/- beta*j1**2`. This also checks the equilibrium flow coefficients.
- Comparison of Gaussian-integrated observables from the original and rotated
  covariance factors at quadrature order 32.
- Independent central finite-difference checks of logweight Jacobian columns.
- Deliberate violations of each scalar equation, invalid covariance rejection,
  evaluation-budget exhaustion, and a misleading optimizer success flag.
- Zero and negative magnetization, zero temperature, nonzero mean coupling,
  non-unit coupling scale, asymmetry, and the existing configuration, output,
  progress, and conservation behaviour.
- A coarse-quadrature failure at `m0=0.6` is retained as an explicit rejection
  test; increasing quadrature to 8 resolves the tested run, agreeing with 12.

The initial version-2 before/after comparison of the successful `t_end=0.001`, 24-cell,
order-4 case found exactly unchanged density and first four trajectory columns.
The fifth column changes from approximately `1.16204e-8` (density RMS) to
`3.07765e-8` (maximum equation error), as intended.

In that version-2 validation, with default physical settings, `m0=0.2`, 24 cells and quadrature order 4, runs
to `t_end=0.01`, `0.1`, and `1` completed in 5, 50, and 500 steps. In the 500-step
run the maximum equation residual across all accepted saddles was approximately
`3.08e-8`, below the default `2e-5` tolerance; mass was conserved and the final
density remained positive. Final magnetization was approximately `0.0822619384`.
An additional final-density solve gave `q` approximately `0.0118704705` and maximum
equation residual approximately `1.04e-15`.

### Short-time refinement study

All rows use `t_end=0.01`, `m0=0.2`, default physical parameters and field bounds
`[-6,6]`. These rows were repeated with version 3 and `saddle_tol=1e-8` to keep
equation errors smaller than the mesh/time differences being compared. q is
evaluated by an additional solve at the common final time.

| Cells | Quadrature | dt_max | Final m | Final-time q |
| --- | --- | --- | --- | --- |
| 24 | 4 | 0.002 | 0.198012434741 | 0.039211208930 |
| 24 | 8 | 0.002 | 0.198012434741 | 0.039211210581 |
| 24 | 12 | 0.002 | 0.198012434741 | 0.039211210581 |
| 48 | 8 | 0.002 | 0.198012682805 | 0.039210304234 |
| 96 | 8 | 0.002 | 0.198012795522 | 0.039210080028 |
| 24 | 8 | 0.001 | 0.198012228173 | 0.039211127125 |
| 24 | 8 | 0.0005 | 0.198012124951 | 0.039211083806 |

All runs conserved mass to floating-point precision and retained positive
densities. The largest accepted equation residual was below `8.6e-9`.
Successive mesh/timestep changes reduce the differences in these observables.
This short-time study does not establish accuracy at longer times.

The historical reproduction procedure was to construct `Parameters(cells=...,
gaussian_order=..., dt_max=..., saddle_tol=1e-8)`, call `model.run(.01, .2)`, then call
`model.saddle(final_density)` for the final-time q. The normal run output retains
the original pre-update q convention.

These table values belong to the earlier spatial integrator. The same calls
with the current solver exercise the current method and are not expected to
reproduce historical trajectories bitwise.

## 3. Long-run objective correction (19 October 2026)

The previous objective (`bounded_least_squares_RS_v1`) reproduced the reported
failure exactly: `t=6.036`, step 3019, after six objective evaluations. The
optimizer reported gtol convergence, optimality approximately `4.22e-15`, and
cost approximately `3.66e-13`. Its conditional Gaussian variance was effectively
zero. The substituted objective's Q0 error was only about `3.37e-8`, while
independently evaluating the original Q0 equation gave `2.01001e-5`, above the
unchanged `2e-5` tolerance.

Repeating that fit with tighter gradient tolerance or with gradient stopping
disabled did not improve it. Thus increasing the evaluation budget or blindly
tightening stopping thresholds was not a verified fix. The objective had reached
a stationary constrained fit and was not measuring the same scalar residuals as
the independent check.

The version-2 objective (retained in `bounded_least_squares_RS_v3`) directly includes all original
scalar equation errors in the objective, with updated analytic logweight
derivatives. Independent residual acceptance, the tolerance, covariance
constraints, and spatial evolution formulae remain unchanged. Failure diagnostics
now include the optimizer termination reason and conditional variance.

At the saved failing state, the new objective passes the unchanged tolerance at
order 4 (maximum error approximately `2.45e-6`), but does not reach an exact root.
Order 8 or 12 closes the same density to errors below `1e-10`. Several alternative
order-4 initial guesses and more evaluations did not establish a more accurate
root. This is evidence of quadrature sensitivity and a constrained-fit limitation,
not a proof that no other order-4 root exists.

Full independent runs from the same initial density give:

| Quadrature | End time requested | Result | Final m | Last recorded q |
| --- | --- | --- | --- | --- |
| 4 | 20.01 | Fails at t=6.682, worst equation R0, error about 2.00122e-5 | — | — |
| 8 | 20.01 | Completes 10006 steps | 0.000125273488 | 0.187086921 |
| 12 | 20.01 | Completes 10006 steps | 0.000125425948 | 0.186819500 |

In both successful runs every accepted saddle satisfies the unchanged tolerance,
total mass differs from one by less than `4e-15`, and all final densities are
positive. Maximum equation residuals are approximately `1.99984e-5` and
`1.99990e-5` respectively: warm starts already within tolerance need not be
optimized further. The roughly 0.14% difference in final recorded q is a useful
quadrature comparison, not evidence of complete continuum convergence. Further
quadrature, mesh, domain and time refinement is needed for physical conclusions.

The parameter file was changed to quadrature order 8 with the user's approval;
its other input settings were retained. Order 8 was already the command-line
default. No automatic quadrature changes, tolerance relaxation, or fallback
square roots were introduced.

`tests/saddle_t6_036.npz` preserves the actual failing state. Tests use it to
check the regression, verify that the optimizer sees the same scalar residuals
as the independent checker, validate the late-time analytic Jacobian, and compare
quadrature orders 8 and 12. The full 20.01 runs were also executed manually;
short and analytic-equilibrium regressions continue to pass. For reproducibility,
the long validation runs set `OPENBLAS_NUM_THREADS=1` to limit numerical-library
thread overhead; the program does not set this environment variable itself.

## 4. Low-temperature numerical-state correction (19 October 2026)

For `m0=0.3`, `temperature=0.01`, `cells=24`, `gaussian_order=8` and the other
default physical/numerical settings, version 2 reproduces the reported failure
at `t=2.466`, step 1234. One normalized chi value underflows to zero while the
target density in that cell remains positive (the smallest target density is
approximately `7.87e-8`). The next warm start computes log(0), passes a negative
infinity to SciPy, and triggers invalid arithmetic warnings and premature xtol
termination. The stored overlap at this point is about 0.1188, so this is not
the singular q=1 limit.

Version 3 stores the finite logweights persistently as `model.logchi`; numerical
calculations no longer exponentiate chi merely to take its logarithm on the next
step. `model.chi` remains a linear inspection view, and assignment to it converts
supplied nonnegative finite weights into logarithms. A genuinely supplied zero
on positive target support produces a clear error before SciPy is called.
The first underflow checkpoint has a smallest finite logweight near -1224.

Keeping logweights alone passes the original failing state, but exposes another
conditioning problem: very weakly determined tails can drift to logweights near
-4e11. Their predicted probabilities and Jacobian columns become effectively
zero. A finite representation alone cannot make the optimizer recover such a
direction.

Before optimization, version 3 therefore performs one log-domain density-matching
initialization step at the warm-start covariance. Write the normalized measure
as `exp(u_i + K_ai) / Z_a`, with Gaussian weights `g_a` and the spatial measure
included in `Z_a`. The new initial logweight is

```text
u_i_new = log(rho_i) - logsumexp_a(log(g_a) + K_ai - log(Z_a)).
```

This computes a finite density-based seed directly from the Gaussian kernel,
avoiding subtraction of enormous negative logweights. It is an iterative
proportional fitting step used only as initialization, not an alternative saddle
equation. No density floor, weight cap, negative-square-root fallback, tolerance
relaxation, or automatic quadrature change is introduced. The independent full
equation checks remain the acceptance criterion, and failed solves do not commit
trial state.

Version 3 also stops optimization as soon as an iterate passes every independent
equation check at `saddle_tol`, using the same criterion as warm-start acceptance.
The previous code could continue for dozens of evaluations after meeting this
criterion, moving weakly identified tail weights to extreme values while seeking
a tighter optimizer gradient threshold. The new internal early-exit mechanism
never accepts an unchecked iterate; final independent checking still occurs.
The saved solver record identifies the stopping rule. Tests cover this exit,
the evaluation budget, and stricter-tolerance checkpoint solves.

Consequently version-3 trajectories need not reproduce version-2 oversolved
trajectories exactly. A per-solve residual tolerance is not a bound on accumulated
trajectory error; tolerance refinement remains necessary.

Regression tests cover the actual underflow checkpoint, invariance of the
normalized measure when every linear chi underflows under a common log shift,
recovery of an artificially flat tail, and rejection of nonfinite optimizer
inputs. A later checkpoint at about t=5.432 retains a strict equation-check
failure at order 8, while orders 12 and 16 close the same density to below 1e-9.
This distinction guards against hiding later resolution/solver limitations when
fixing the original numerical-state bug.

The high-temperature reference (`temperature=1.5`, `m0=0.2`, order 8,
t_end=20.01) still completes 10006 steps under the final version-3 stopping rule.
Its final magnetization is approximately 0.000124613939196, recorded q is
approximately 0.187094894627, mass differs from one by less than 1e-14, and all
accepted equation errors remain below 2e-5. The small differences from the
version-2 numbers above reflect the changed stopping rule, not a change of the
equations or an assertion of global trajectory accuracy.

Full low-temperature runs use exactly the reported physical parameters:
temperature 0.01, m0=0.3, j0=0, j1=1, asymmetry=0, cells=24, t_end=10.01.
Only quadrature order varies; all other numerical settings retain their defaults.
RuntimeWarnings are treated as errors in these checks.

| Quadrature | Full-run result with version 3 |
| --- | --- |
| 8 | Equation check fails at t=5.430, worst R0 approximately 2.09048e-5 |
| 12 | Equation check fails at t=6.982, worst R0 approximately 2.08983e-5 |
| 16 | Equation check fails at t=8.396, worst R0 approximately 2.18290e-5 |
| 24 | Completes 5005 steps to t=10.01 without warnings |
| 32 | Completes 5005 steps to t=10.01 without warnings |

The failed higher-order fits do not prove nonexistence of another root. They
demonstrate that fixing the original underflow does not justify suppressing later
equation-check failures or assuming that order 8 resolves this low-temperature
trajectory.

For order 24 the final m is approximately 0.0610504275085, the last recorded q
0.602342923950, and the final energy -0.731327878778. For order 32 these values
are approximately 0.0610468712183, 0.595811693920, and -0.731345443672. Both final
densities are positive, mass differs from one by less than 1e-14, and the maximum
accepted equation residual remains below the unchanged 2e-5 tolerance. The final
recorded q differs by about 1.1% between these orders: the successful runs are not
evidence of full quadrature or physical convergence. Grid, field-domain, timestep,
tolerance and replica-stability checks remain necessary. No physical parameter
or the user's input quadrature is changed automatically.

## 5. Asymmetry and the fully asymmetric endpoint

`asymmetry` is k in equation (1.18), not the reciprocal-coupling correlation.
The project accepts 0 <= k <= 1 in both solvers and the combined input reader.
The simulator uses j1*(S+k*A)/sqrt(N*(1+k*k)), with unit normal S and A; thus
reciprocal disorder correlation is (1-k*k)/(1+k*k).

At k=1, (1-k*k)*Theta1 vanishes. The Gaussian cross-loading b is fixed to zero,
and the corresponding optimization coordinate is Theta1 itself rather than b.
Its defining original saddle equation remains in the objective and independent
checks. The scalar Q0 equation, covariance positivity and all density equations
are retained. The logweight analytic Jacobian remains valid because Theta1 is an
independent coordinate; covariance-coordinate derivatives remain numerical.
This removes the coordinate singularity without approximating k or relaxing
any tolerance. The interior 0<=k<1 parameterization is unchanged.

Endpoint tests at T=0, 0.01 and 1.5 reach t=0.1 with all equation checks passing,
normalized nonnegative densities, a reconstructible committed measure and
m(t)=m0*exp(-t) for j0=0 and the initially even total field density. The combined
runner is tested with k=0 and k=1, invoking both existing command-line programs.
These tests do not imply convergence for arbitrary duration or initial conditions.

Additional endpoint runs at T=0, 0.01 and 1.5, each with j0=0 and j0=1,
completed t_end=1 (500 steps) using 24 cells and quadrature order 24.
All accepted equation residuals remained below 1.5e-6 (tolerance 2e-5).
Short before/after comparisons at k=0, 0.3 and 0.8, with j0=0.4, j1=1.2,
24 cells and quadrature order 8, produced bitwise-identical trajectories and
final densities. The historical regression suite also remains passing.

## 6. Strong-mean-coupling stability and efficiency

The following sequence predates the exponential-fit spatial integrator. Here
"final" means the final safeguarded saddle implementation at that stage, not
an assertion that these forward trajectories have been rerun after the flux fix.
Single-update and unconditional four-update initialization are intermediate
versions; the current initializer selects the best coupled objective among the
four proposals and the untouched warm start.

The two reported combined-run failures were reproduced with their original
settings: temperature=0.01, j0=2, j1=1, m0=0.3, cells=64, Gaussian order 32,
dt_max=0.002, saddle_tol=2e-5 and saddle_maxiter=120. With the previous transport
normalization, k=0 failed at t=5.11564541, step 19583, density residual
1.38851e-4; k=1 failed at t=6.80527645, step 4887, density residual 2.22261e-5.
Both exhausted the evaluation budget. Target density and last accepted state
are retained under `tests/saddle_ferromagnetic_*.npz`.

### Diagnosis and limited correction

The absolute density tolerance does not bound relative error in tiny tail cells.
Phi3 is linear in saddle-weighted moments. The old transport divided those fitted
moments by the evolving density rather than their fitted marginal. At the k=1
checkpoint, a fit accepted after 207 evaluations (budget raised experimentally
to 600) had a tail marginal about 1.06e-5 against an evolving density about
1.25e-7. This was within the unchanged 2e-5 absolute density tolerance. Dividing
by the evolving density produced maximum face speed about 1976.6 and CFL step
3.79e-5. Normalizing by the fitted marginal instead gave maximum speed about
9.18 and permitted the configured 0.002 timestep.

The implementation now forms conditional transport moments by dividing Phi3 by
its own fitted marginal, and forms the flux by multiplying by the evolving
rho. At an exact saddle this is the same expression. The existing 1e-12
transport density floor, zero-flux boundaries, upwind scheme, spin-flip step,
saddle equations, least-squares objective and independent tolerance checks are
unchanged. No target density or residual is clipped. Zero fitted density gives
zero conditional transport to avoid 0/0.

Larger budgets alone were not a sufficient fix: budget 600 accepted each original
failed state (about 255 evaluations for k=0 and 207 for k=1), but the old k=1
transport failed again two steps later. Broad Jacobian scaling worsened the
forward time-step slowdown and was rejected. Order 64 closed both original
checkpoints within 120 evaluations, but the old tail normalization still imposed
very small timesteps. On the corrected trajectory, order 32 still failed the k=1
saddle at t=6.77, density residual about 2.10104e-5. Thus quadrature resolution
remains an independent limitation; the correction does not bypass that rejection.

A second, independent numerical issue occurred during the k=0 corrected-order-32
trial: the default LAPACK SVD failed on a finite 264-by-130 trust-region matrix
(entries approximately -78 to 216). An alternate LAPACK driver solved that matrix,
and a uniform half-scale coordinate change allowed the existing TRF optimizer to
close the same saved state. The implementation uses only the public least_squares
API: on an actual SVD-convergence exception it retries once from the best finite
checked point with x_scale=0.5, using only the remaining original evaluation budget.
Other linear-algebra exceptions are not swallowed. The nonlinear algorithm,
objective, bounds and acceptance checks are unchanged; normal runs retain their
original scaling. The rare SVD state is retained as a portable fixture (some
LAPACK versions solve it directly). NPZ metadata records `svd_recoveries`.

### Historical forward comparisons

Both existing long-run examples complete with every accepted equation checked:

| Case | Steps, before/after | Maximum change in m over the run | Maximum change in E | Change in final recorded q |
| --- | --- | --- | --- | --- |
| T=1.5, j0=0, k=0, m0=0.2, 24 cells, order 8, t_end=20.01 | 10006 / 10006 | 1.10e-6 | 1.46e-6 | -2.61e-5 |
| T=0.01, j0=0, k=0, m0=0.3, 24 cells, order 24, t_end=10.01 | 5005 / 5005 | 1.16e-8 | 2.59e-7 | -9.64e-6 |

Mass remains one to rounding, densities stay nonnegative, and all accepted
residuals remain at or below 2e-5. Comparisons interpolate the old m/E onto the
new saved times; q retains its documented pre-update convention. These checks
establish numerical regression behavior, not continuum or replica-symmetry
accuracy and not agreement with the thesis figures.

### Algebraically equivalent Gaussian evaluation

The higher quadrature order makes repeated full-tensor normalization expensive.
For fixed spin s and field-noise node z1, define

```
A_s(h,z1) = exp(u_s(h) - x*x/(2*j1*j1*(1-q)) + lambda*x*z1)
Z_s(z1) = dh*sum_h A_s(h,z1)
```

The normalized measure factors exactly into the field conditional A_s/Z_s and
the two-spin mixture proportional to Z_s*exp(s*(b*z1+sqrt(v)*z2)). Computing
these normalizations separately retains the full tensor-product quadrature and
the same observables, equations and independent checks. Shifted logarithms avoid
linear underflow; zero-support spin populations contribute zero. Direct comparison
against the original logsumexp/kernel formula covers q=0 through 0.99, nonzero
cross-loading/variance, logweight shifts of -1200 and an empty spin population.
The measures agree to rounding and conditional normalization is preserved.
A local 64-cell/order-64 benchmark made the Gaussian measure evaluation about
nine times faster; this is not a claim of a ninefold speedup for a complete run.

During the order-64 full-tensor check, k=1 completed t_end=8 with budget 120
(4001 steps, final m=0.9205561750 and E=-0.9265932155). However, k=0 reached
t=6.328 before exhausting that same budget, with density residual 2.18266e-5.
The final verification therefore uses the same order 64 together with the
existing, explicitly configured saddle_maxiter=600 control. Physical settings
and saddle_tol=2e-5 are retained. The factored measure evaluation avoids much
of the repeated normalization cost; the higher budget is a cap, not a requirement
to perform 600 evaluations on already converged saddles.

### Fixed-covariance initialization refinement

The single matching update was still a bottleneck after the transport correction.
At the saved k=0 state reached at t=6.516 with 64 cells and order 64, the coupled
fit failed after 600 evaluations. Raising its budget to 1500 closed the same state
only after about 1100 evaluations. Orders 96 and 128 did not close it within 120
evaluations either; order 128 took about 390 with budget 600. A cold covariance
restart failed and was rejected. These results do not support treating it as only
a quadrature-resolution problem.

Repeating the existing log-domain matching update four times at the ORIGINAL
warm-start covariance closed that state in 19 coupled evaluations. The original
order-32 k=0 failure state also closed within 120, whereas the original order-32
k=1 state still failed its unchanged checks. Thus no formerly rejected state is
silently accepted: each accepted result is rechecked against all original equations.
The t=6.516 state is retained in `tests/saddle_ferromagnetic_k0_t6_516.npz`.

The matching update u <- u + log(rho) - log(fitted_density) is evaluated from
factored log marginals, without exponentiating small probabilities. Its equivalence
to the original log-kernel expression is independently tested, including a common
logweight shift of -1400. Four updates refine logweights only, at fixed covariance;
the same bounded least-squares solver then fits the coupled variables.

The historical forward comparison table above includes the final safeguarded matching
initialization. Final recorded q changes more than m/E (about 9.64e-6 in the
low-temperature example). This illustrates that a 2e-5 equation tolerance is not
a bound on q or accumulated trajectory error. It must be refined as part of
scientific convergence studies.

At fixed saved density `saddle_t6_036.npz`, quadrature order 12 and tolerance
1e-11, the pre-change and final solvers both converge in 11 coupled evaluations
with all residuals below 1e-14. Their q, mu, Theta0, Theta1 and Q0 differ by at
most 1.61e-15. This checks that the initialization/evaluation changes retain the
same well-resolved saddle root, rather than introducing a new set of equations.

### Safeguard matching with the original least-squares merit

Four unconditional matching updates still failed the strong-mean k=0 forward
run at t=6.736 after 600 evaluations (density residual 2.05924e-5). A separate
prototype that rejected and halved the preceding timestep failed eight successive
retries, down to dt=7.8125e-6. Timestep backtracking was therefore rejected rather
than incorporated into the production integrator. The exact failed state is in
`tests/saddle_ferromagnetic_k0_t6_736.npz`.

The density-matching proposals improved the marginal but could worsen the
COUPLED objective. At this saved state, retaining the better original continuation
guess closed the unchanged equations in 11 evaluations. The final initialization
computes four matching proposals at the unchanged warm covariance, and selects
the lowest existing objective norm among all proposals AND the untouched warm
start. This does not rescale residuals or relax any acceptance test.

With this safeguard, the two original order-32 failure states both close within
120 evaluations (about 8 and 10 in the focused probes). The tail-velocity regression
uses a separately frozen historical accepted approximation, so improvements in
initialization cannot remove the actual numerical-error scenario being tested.

### Final forward-validation outcome and remaining limit

The final safeguarded implementation was checked from t=0 with the original
physical settings and field domain. Neither original order-32/120-evaluation
run completed: k=0 failed at t=5.534 and k=1 at t=6.778. At order 64 with budget
600 (same 64 cells, dt_max=0.002 and saddle_tol=2e-5):

| k | Outcome | Accepted steps | Final m | Final E | Maximum accepted residual |
| --- | --- | --- | --- | --- | --- |
| 1 | Completes t=8 | 4001 | 0.9205563884 | -0.9265931712 | 1.99841e-5 |
| 0 | Fails at t=6.552 | 3276 before failure | — | — | Failed density equation 3.144e-5 |

The completed k=1 run preserves mass to rounding, has positive final density and
uses at most 15 coupled objective evaluations for any accepted saddle. Its budget
600 is therefore not responsible for success; the cap 120 would not be reached
on that validated path. No SVD recovery was required. The remaining k=0 failure
is not declared solved, and its final state is retained in the diagnostic output
folder. User parameter files and the original sweep manifest are unchanged.

The historical comparison subset was saved at
`Output/stability_validation_2026-10-19_200425_292139/manifest.json`. It reused
the original Monte Carlo results for sets 3 and 6, included the completed k=1
archive and explicitly marked k=0 failed. The associated logs, reproduction
parameters, checkpoint, summary and PNGs documented that diagnostic configuration,
not convergence of both methods for both sets. **This folder is absent from the
project directory as checked on 25 October 2026**, so the former instruction to
reopen that manifest is no longer applicable to this checkout. The regression
fixtures in `tests` remain available; the numerical results above are retained
as historical records, not newly reverified outputs.

### Field-domain diagnostic (not a forward-run fix claim)

For j0=2 a large magnetization shifts the field mean toward the upper boundary.
The independent-spin initial Gaussian is an analytic saddle with q=m*m in the
infinite-domain limit. Checking that state at resolved quadrature order 128 and
the SAME field-cell width 0.125 isolates finite-domain effects:

| m | Maximum equation error, [-6,6] | Maximum equation error, [-8,8] |
| --- | --- | --- |
| 0.3 | 3.64e-7 | 9.90e-13 |
| 0.9 | 1.58e-4 | 4.26e-9 |
| 0.95 | 6.22e-4 | 1.81e-8 |

The existing [-6,6] truncation can therefore matter well above saddle_tol for
large mean fields, even before considering time integration. This is evidence
for a domain-convergence study, not proof that widening the bounds cures the
full failing trajectory. A candidate [-8,8] domain would need at least 86 cells
to retain the original 12/64 cell spacing, and quadrature refinement should be
checked separately. No such changed-domain forward run has been validated here.
All original equation checks remain active; tolerance is not increased to hide
these errors.

## 7. Equilibrium-preserving field transport (23 October 2026)

This investigation follows the saddle work above and changes the spatial
integrator. Its benchmarks therefore supersede earlier upwind results for the
reported high-temperature equilibrium case.

### Diagnosis and scope

The reported symmetric-model case is `j0=0.75`, `temperature=1.5`, `j1=1`.
Thesis (6.19)-(6.20) gives `m=q=0`, `E=-1/3` and

```
rho_sigma(h) = exp(-(h - sigma/T)^2/2) / (2*sqrt(2*pi)).
```

The saved 128-cell runs in
`Output/comparison_2026-10-23_063455_095996` instead converge near `-0.36513`.
Independent energy reconstruction from their final densities reproduces the
saved values. The formula `E=-dh*sum(h*(rho_plus-rho_minus))/2` is correct.
The simulated 30-path energy means over t=100-150 are approximately -0.333819
and -0.333055 for m0=-0.02 and 0.4 respectively; these paths share disorder.

With the analytic equilibrium as input, the saddle and continuum coefficients
pass at machine precision. The old upwind-drift/central-diffusion flux nevertheless
moves that density. On 128 cells over [-8,8], its L1 density-change rate approaches
0.048248 as dt decreases. At dt=0.002, 0.0002 and 0.00002, measured rates are
0.048208, 0.048244 and 0.048248. This isolates a spatial error, not a timestep,
energy-output or equilibrium-reference error. The old equilibrium test checked
coefficients but did not advance the density.

### Numerical change

For face velocity v, cell width dh and diffusion D, use

```
P = v*dh/D
B(x) = x/(exp(x)-1)
J_face = D/dh * (B(-P)*rho_left - B(P)*rho_right).
```

This Scharfetter-Gummel flux fits the constant-coefficient drift/diffusion
solution on each interval. Its local zero-current ratio is exp(P).
For the Gaussian equilibrium above, the arithmetic average of the linear drift
gives the exact log-density ratio on the grid. Thus transport and spin flips
separately preserve this equilibrium, to roundoff and field truncation error.
No equilibrium distribution or energy target is supplied to the integrator.

The implementation evaluates nonnegative left/right transfer rates with a
small-P series and nonpositive exponentials. For D=0 it uses the pure-upwind
limit. The explicit timestep is bounded by 0.8 divided by the maximum total
outgoing rate per cell. The update is expressed as retained mass plus incoming
mass, with zero flux through the domain boundaries. No clipping or
renormalization is introduced.

The saddle equations, coefficient construction, density floor, energy formula,
initial conditions, simulator and reference solver are unchanged. New archives
record the spatial flux and timestep rule in their solver JSON. No input syntax
or default parameter values change. Historical archives are not rewritten.

### Full-flow benchmark

All rows below evolve from the random-spin initial distribution with m0=0 to
t=30, using [-8,8], Gaussian order 4, dt_max=0.002 and saddle_tol=1e-9.
Gaussian integration is degenerate on the symmetric m=q=0 branch, so order 4
is sufficient for this particular check. These are full calls to SpinGlass.run,
including a saddle solve at every step, not frozen-coefficient evolutions.

| Cells | Old upwind energy | Exponentially fitted energy |
| --- | --- | --- |
| 64 | -0.397300367438 | -0.333333331807 |
| 128 | -0.365130061682 | -0.333333332041 |
| 256 | -0.349189904539 | -0.333333332094 |

All three corrected runs retain nonnegative densities, mass within 3e-15 of
unity, and every accepted saddle residual below 1e-9. New trajectories and
summary results are saved in `Output/flux_validation_2026-10-23_074857`.

Halving dt_max to 0.001 in the 128-cell m0=0 benchmark gives
E=-0.333333332034 at t=30, differing by 6.9e-12 from dt_max=0.002.
Its mass, positivity and saddle checks also pass. This final-time comparison
does not eliminate timestep error along the transient.

For example, at t=5 the corrected m0=0 energies on 64, 128 and 256 cells
are -0.326968053, -0.327115290 and -0.327153057. At 128 cells, halving the
timestep changes the t=5 energy by 4.64e-6. These finite transient differences
remain even though all grids approach the same equilibrium.

The two reported initial magnetizations were also checked to t=30 at 128 cells,
Gaussian order 8, dt_max=0.002 and the normal saddle_tol=2e-5:

| m0 | Final energy | Final m | Maximum accepted residual |
| --- | --- | --- | --- |
| -0.02 | -0.333332785167 | -0.000021559784 | 1.99994e-5 |
| 0.4 | -0.333331899646 | 0.000415105831 | 1.99998e-5 |

Both conserve mass within 2e-15 and retain nonnegative density. The original
order-64 archives give -0.365128793 and -0.365127533 respectively at t=150.
The corrected checks use order 8, not the original order 64; their completion
and agreement with the high-temperature energy are not a general quadrature
convergence claim.

### Regression coverage and limits

At the flux-fix validation, `python -m pytest -q` reported **181 passed**.
That recorded run exercised the existing regression suite plus new checks:

- Advance the analytic SK equilibrium for 100 steps at multiple grid sizes and
  temperatures, including j1=1.3. Density and energy stay at roundoff accuracy.
- Check local zero-current balance for both drift signs and very small Peclet
  numbers; check finite nonnegative rates for large Peclet numbers and D=0.
- Stress conservation and positivity with large converging drift and diffusion.
- Evolve a random-spin state self-consistently to t=20 on a deliberately coarse
  32-cell grid; energy reaches -0.333332326, within 2e-6 of -1/3.
- Check saved method identifiers along with the existing archive checks.

An extra m0=0.4, order-8 trial with deliberately tightened saddle_tol=1e-9
stopped at t=0.002: Q0 residual 2.86833e-9 exceeded the requested tolerance after
gtol termination. That trial is recorded as failed; the saddle solver and its
acceptance checks were not changed to make it pass.

Exact preservation of this Gaussian equilibrium does not prove exact preservation
of every non-Gaussian SK equilibrium. Low-temperature, asymmetric and strongly
magnetized long trajectories still require refinement and physical validation.
Passing regression fixtures does not establish long-time convergence in those
regimes. The original t=150, order-64 production sweep has not been rerun.

## 8. Current interpretation and remaining limits

This validates solution of the discretized real-Gaussian RS equations in the
tested cases, not global existence or uniqueness of their roots. A least-squares
failure is not proof of physical inadmissibility. Near `q=1` the representation
is singular; fully polarized states are explicitly rejected. Low quadrature
order, finite field boundaries, grid spacing and the spatial flux approximation
can affect results even when the saddle residuals are small. The covariance
rotation also changes finite-order quadrature errors.

The saddle-stage investigation did not change the spatial drift/diffusion or
spin-flip formulas. Section 7 subsequently replaced their spatial discretization
and added a targeted Monte Carlo comparison for the reported high-temperature
case. Neither stage establishes reproduction of all thesis figures, general
Monte Carlo agreement, field-domain convergence or replica-symmetry-breaking
stability.

The current implementation retains independent equation checks, persistent
logweights, safeguarded four-proposal initialization, bounded SVD recovery and
the factored Gaussian measure, together with the Scharfetter-Gummel transport
and outgoing-rate timestep bound. It does not change the energy definition or
upgrade the closure from RS to full RSB. Later scientific runs must still state
their grid, field domain, quadrature, timestep, saddle tolerance and solver
identifiers. The high-temperature flux fix is not a demonstrated cure for all
previous low-temperature or strong-mean-coupling failures.
