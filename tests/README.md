# Long-run saddle regression checkpoint

`saddle_t6_036.npz` contains the target density and last accepted saddle state
immediately before the reported failure at `t=6.036`, step 3019. It was generated
with the previous target-moment objective, using `cells=24`, `gaussian_order=4`,
`dt_max=0.002`, `m0=0.2`, and the default physical parameters (temperature 1.5,
j0=0, j1=1, asymmetry=0, field bounds -6 to 6).

The arrays record rho and chi, and scalar q, mu, theta0, theta1, r, rterm, tterm.
The checkpoint permits testing the actual late-time failure without repeating
3018 timesteps in every test. This is a numerical regression fixture, not an
independently validated physical reference trajectory.

`saddle_low_temperature_t2_466.npz` records the positive target density and finite
logweights immediately before the reported low-temperature failure (`m0=0.3`,
temperature 0.01, cells 24, quadrature 8, other defaults). It was captured with
the log-state correction alone, before the density-matching initialization was
added. Its smallest logweight is approximately -1224, which underflows to zero
in linear form. The previous solver lost this information and passed a negative
infinity to SciPy on the next step. The scalar saddle fields are saved alongside
rho and logchi.

`saddle_low_temperature_t5_432.npz` records a later state obtained while
investigating this case. It contains rho, linear chi (all positive in this
snapshot), and the scalar saddle fields. Tests verify that order 8 retains a
genuine equation-check rejection while orders 12 and 16 close this same density
accurately; this guards against removing admissibility or accuracy checks just
to continue a run.

## Strong-mean-coupling sweep failures

`saddle_ferromagnetic_k0_t5_115645.npz` and
`saddle_ferromagnetic_k1_t6_805276.npz` reproduce the two reported failures
from the 12-case combined sweep: temperature=0.01, j0=2, j1=1, m0=0.3,
64 cells, quadrature order 32, dt_max=0.002, saddle_tol=2e-5, budget 120.
The original evolving-density normalization is used to generate both snapshots.
They capture the target density and last accepted logchi/multiplier state at
t=5.11564541 (step 19583, k=0) and t=6.80527645 (step 4887, k=1).
Both failures were reproduced with the same reported residuals before initialization refinement. Effective
Parameters are stored in JSON alongside rho, logchi and the scalar warm-start fields.

The transport fix does not change the saddle equations, so with the original one-update initialization these target
densities fail their original 120-evaluation checks. Four matching updates now
close the k=0 state within that budget, while the k=1 state still requires
resolution or budget refinement. With budget 600, the
k=1 state can be accepted after about 207 evaluations, while the old transport
then fails again two steps later. The accepted approximate saddle is used to test
that its relative tail error cannot generate an artificial CFL restriction. These
are numerical regression states, not validated physical reference data.

`saddle_ferromagnetic_k0_svd.npz` is a separate state encountered during the
corrected-transport, order-32 k=0 investigation. Its finite trust-region matrix
triggered LAPACK's default SVD nonconvergence in the development environment.
The regression allows either direct convergence or the bounded one-time
coordinate-rescaling recovery, depending on LAPACK version, and requires all
original equation checks to pass. It has the same array schema as the two
original failure states.

`saddle_ferromagnetic_k0_t6_516.npz` is the later k=0 target and last accepted
state after correcting transport, with 64 cells/order 64 and a 600-evaluation cap.
It still failed with one matching initialization step. Four fixed-covariance
matching updates close the same density within 120 evaluations; this is a focused
regression for initialization robustness, not an independently known physical solution.

`saddle_ferromagnetic_k0_t6_736.npz` captures the later failure of unconditional
four-step matching at order 64. Safeguarding the original objective norm makes
this density close within 120 evaluations. The same safeguard also closes both
ORIGINAL order-32 failure snapshots within 120 evaluations; earlier descriptions
of their rejection refer to the historical initialization.

`saddle_ferromagnetic_k1_accepted_tail_error.npz` freezes the accepted approximate
saddle used in the tail-velocity regression. It was obtained from the original
k=1 failure with the historical initializer and budget 600. Its target density
and saddle still pass the original 2e-5 checks, but differ greatly in relative
terms in a tiny tail cell. It isolates the transport correction from subsequent
initial-guess improvements.
