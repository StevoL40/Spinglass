"""Research reimplementation of Laughton (1995), Appendix D.

The replica-symmetric saddle closure is reconstructed from the printed C source.
This is an experimental implementation, not a numerically validated reproduction.
"""
from __future__ import annotations

# Standard-library support for settings, command-line input, timing and files.
from dataclasses import asdict, dataclass
from datetime import datetime
import argparse
import json
import math
import time
from pathlib import Path
# Numerical arrays, Gaussian quadrature and the nonlinear saddle solver.
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.optimize import least_squares


# Reading guide
# -------------
# This script evolves a joint spin/local-field density, not individual spins.
# Thesis (6.15) supplies a replica-symmetric (RS) saddle for the current density;
# (6.17) supplies its transport coefficients; (6.16) supplies the evolution law.
# The time loop is therefore: saddle -> coefficients -> transport/spin flips.
# Solving the saddle accurately is necessary, but does not establish continuum
# accuracy or the physical validity of RS in a glassy regime.
#
# Array and notation conventions used throughout:
#   rho[s, h]       density per unit field; s=0 means +1, s=1 means -1.
#   measure[s,i,j,h] conditional spin/field density at two Gaussian nodes i,j.
#   dh*sum(rho)     total probability, which should stay equal to one.
#   gw[i,j]         normalized tensor-product Gaussian integration weights.
#   theta0, theta1  thesis R0, R1; r**2 is the auxiliary multiplier Q0.
#   js, ok, ko      J1**2, (1-k**2), (1-k**2)/(1+k**2), respectively.
# The q and Q0 multipliers are different quantities. Neither is advanced by an
# independent time equation here: the saddle determines them from the density.
#
# Numerical kernels and saddle machinery come first; coefficients, evolution,
# observables and the time loop follow. Parameter-file/CLI and archive handling
# are deliberately at the end. README.md explains the equations; the detailed
# failure investigations and verification limits are in NUMERICAL_INVESTIGATIONS.md.
# Run from the project root:
#   python python/spinglass.py config/parameters.txt


@dataclass(frozen=True)
class Parameters:
    """Physical parameters and numerical resolution for direct Python callers.

    Fields are in the same units as J0, J1 and temperature (Boltzmann k=1).
    gaussian_order is the order of EACH auxiliary Gaussian axis, not the total
    node count. saddle_tol bounds independently checked dimensionless equation
    residuals; it is not an error bound on a whole trajectory. dt_max is a cap:
    transport stability can require a smaller step. The CLI selects its own
    cells/quadrature defaults in main() before applying file and CLI overrides.
    """
    # Physical model: mean coupling, disorder scale, temperature and asymmetry.
    j0: float = 0.0
    j1: float = 1.0
    temperature: float = 1.5
    asymmetry: float = 0.0
    # Field-grid and Gaussian-integration resolution.
    hmin: float = -6.0
    hmax: float = 6.0
    cells: int = 120
    gaussian_order: int = 12
    # Nonlinear-solve accuracy/work limits and maximum evolution time step.
    saddle_tol: float = 2e-5
    saddle_maxiter: int = 120
    saddle_damping: float = 0.12  # Legacy input; inactive in the coupled solver.
    dt_max: float = 0.002

    def __post_init__(self):
        # Reject nonfinite inputs before they reach numerical array operations.
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError('All parameters must be finite')
        # Enforce the allowed physical ranges and positive numerical controls.
        if self.saddle_tol <= 0 or self.saddle_maxiter < 1 or self.dt_max <= 0:
            raise ValueError('Require saddle_tol>0, saddle_maxiter>=1 and dt_max>0')
        if self.j1 <= 0 or self.temperature < 0:
            raise ValueError('Require j1>0 and temperature>=0')
        if not (0 <= self.asymmetry <= 1):
            raise ValueError('Require 0<=asymmetry<=1 (thesis parameter k)')
        if self.cells < 8 or self.hmax <= self.hmin or self.gaussian_order < 2:
            raise ValueError('Invalid grid or quadrature')


class SpinGlass:
    """Field grid, continuation state and self-consistent RS evolution.

    The evolving density is passed explicitly. Attributes such as logchi, q and
    theta0 hold the last accepted saddle and seed the next solve. A trial fit
    must not overwrite them until every independent equation check passes.
    """
    @property
    def chi(self):
        """Linear weights for inspection; numerical calculations use logchi."""
        return None if self.logchi is None else np.exp(self.logchi)

    @chi.setter
    def chi(self, value):
        # None clears the continuation weights before a fresh initialization.
        if value is None:
            self.logchi = None
            return
        # Accept a linear inspection/fixture value, but store its logarithm.
        # A supplied zero becomes -infinity; do not invent a positive floor.
        value = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError('chi must be finite and nonnegative')
        with np.errstate(divide='ignore'):
            self.logchi = np.log(value)

    def __init__(self, p: Parameters):
        """Build a cell-centred field grid and standard-normal quadrature."""
        # Set the fixed grid geometry used by every density and field integral.
        self.p = p
        self.dh = (p.hmax - p.hmin) / p.cells
        self.h = p.hmin + (np.arange(p.cells) + .5) * self.dh
        # Build both auxiliary integration axes and their joint node weights.
        x, w = hermgauss(p.gaussian_order)
        # hermgauss integrates exp(-x**2); rescale to expectation under N(0,1).
        self.z = np.sqrt(2.) * x
        self.w = w / np.sqrt(np.pi)
        self.z1, self.z2 = np.meshgrid(self.z, self.z, indexing='ij')
        self.gw = self.w[:, None] * self.w[None, :]
        # Cache repeated coupling factors, then clear the accepted saddle state.
        self.js = p.j1 ** 2
        self.ok = 1 - p.asymmetry ** 2
        self.ko = self.ok / (1 + p.asymmetry ** 2)
        self.chi = None
        self.q = None
        self.theta0 = self.theta1 = self.mu = self.r = 0.
        self.rterm = self.tterm = 0.

    def initial(self, magnetization: float = .2) -> np.ndarray:
        """Random-spin initial density, thesis (6.18), and its saddle seed.

        Spin fractions are (1 +/- m0)/2 and the independent field is Gaussian
        with mean J0*m0 and variance J1**2. Normalize on the finite grid once;
        this initial truncation normalization is not applied after time steps.
        Resetting here also means each call to run() starts a fresh trajectory.
        """
        # Check the requested population before forming its two density rows.
        if not np.isfinite(magnetization) or abs(magnetization) > 1:
            raise ValueError('|magnetization| <= 1 required')
        # Build the two spin populations on the same initial Gaussian field.
        gauss = np.exp(-.5 * ((self.h - self.p.j0 * magnetization) / self.p.j1) ** 2)
        rho = np.stack(((1 + magnetization) * gauss / 2,
                        (1 - magnetization) * gauss / 2))
        rho /= self.dh * rho.sum()
        # Seed a fresh saddle and clear diagnostics from any preceding run.
        self.chi = np.broadcast_to(np.array([(1 + magnetization) / 2,
                                             (1 - magnetization) / 2])[:, None], rho.shape).copy()
        self.q = magnetization ** 2
        self.theta0 = self.theta1 = self.mu = self.r = 0.
        self.rterm = self.js*self.q
        self.tterm = 0.
        self.last_saddle_residuals = None
        return rho

    def _gaussian_kernel(self, m, q, b, variance):
        """Real Gaussian measure with a field pivot, regular at Q0 = r^2 = 0.

        Field noise is lambda1*z1; spin noise is b*z1 + sqrt(variance)*z2.
        Thus Q0=b^2+variance and (1-k^2)*Theta1=lambda1*b. This is a
        factorization of the covariance in section 6.4.1, not a radicand fallback.
        Deterministic spin terms are absorbed into separate logweights.
        The returned unnormalized exponent has shape (2, order, order, cells).
        This full-tensor expression is retained as an independent test reference;
        the production evaluator below factors its normalization for efficiency.
        """
        if not 0 <= q < 1 or variance < 0:
            raise ValueError('Require 0<=q<1 and nonnegative conditional variance')
        x = self.h - self.p.j0*m
        spin = np.array([1., -1.])[:, None, None, None]
        lambda1 = np.sqrt(q)/(self.p.j1*(1-q))
        return (-x**2/(2*self.js*(1-q)) + lambda1*x*self.z1[None, :, :, None]
                + spin*(b*self.z1 + np.sqrt(variance)*self.z2)[None, :, :, None])

    def _gaussian_log_parts(self, m, q, logweights, b, variance):
        """Factor log probabilities into field-given-spin and spin-mixture parts.

        Return shapes (2, order, cells) and (2, order, order). The first part
        sums to one over FIELD CELLS, before converting to density with 1/dh;
        the second sums to one over SPINS at each Gaussian-node pair. Keeping
        normalization in logs avoids overflow and preserves very small weights.
        An absent spin population has log probability -infinity, not a floor.
        """
        # At fixed spin and field-noise node, the field conditional does not
        # depend on the second Gaussian node. Normalize these two pieces
        # separately instead of repeating exp/logsumexp over the full 4-D array.
        # This is the SAME tensor-product quadrature, not a reduced-order rule.
        if not 0 <= q < 1 or variance < 0:
            raise ValueError('Require 0<=q<1 and nonnegative conditional variance')
        x = self.h-self.p.j0*m
        z = self.z1[:, 0]
        lam = np.sqrt(q)/(self.p.j1*(1-q))
        field = (logweights[:, None, :]-x[None, None, :]**2/(2*self.js*(1-q))
                 +lam*z[None, :, None]*x[None, None, :])
        peak = np.max(field, axis=2, keepdims=True)
        # A wholly absent population has peak=-inf. A finite neutral shift
        # lets it remain absent without forming the indeterminate -inf-(-inf).
        peak = np.where(np.isfinite(peak), peak, 0.)
        shifted = field-peak
        logsum = logsumexp(shifted, axis=2, keepdims=True)
        normalized = np.subtract(shifted, logsum, out=np.full_like(shifted, -np.inf),
                                 where=np.isfinite(logsum))
        spin = np.array([1., -1.])[:, None, None]
        mixture = ((peak+logsum)[:, :, 0, None]
                   +spin*(b*self.z1+np.sqrt(variance)*self.z2)[None, :, :])
        # Normalize the two spin alternatives separately at each node pair.
        mixture -= np.max(mixture, axis=0, keepdims=True)
        return normalized, mixture-logsumexp(mixture, axis=0, keepdims=True)

    def _gaussian_measure(self, m, q, logweights, b, variance):
        """Return M[s,i,j,h], with dh*sum_{s,h} M = 1 at every node pair."""
        field, spin = self._gaussian_log_parts(m, q, logweights, b, variance)
        return np.exp(spin)[:, :, :, None]*(np.exp(field)/self.dh)[:, :, None, :]

    def _gaussian_log_density(self, m, q, logweights, b, variance):
        """Return log rho[spin,field] using the same two-axis quadrature.

        Integrate the spin mixture over the second Gaussian axis first, then
        combine it with the field conditional and integrate the first axis.
        Remaining in logs preserves weights too small for linear exponentiation;
        this marginal is used for density-matching initialization, not evolution.
        """
        field, spin = self._gaussian_log_parts(m, q, logweights, b, variance)
        spin_mean = logsumexp(spin+np.log(self.w)[None, None, :], axis=2)
        return (logsumexp(field+(spin_mean+np.log(self.w)[None, :])[:, :, None], axis=1)
                -np.log(self.dh))

    def _measure(self, m: float, q: float) -> np.ndarray:
        """Evaluate the real RS measure without clipping inadmissible covariance."""
        # Recover the Gaussian coordinates from the stored physical multipliers.
        lambda1 = np.sqrt(q)/(self.p.j1*(1-q)) if 0 <= q < 1 else np.nan
        cross = self.ok*self.theta1
        if not np.isfinite(lambda1) or (lambda1 == 0 and cross != 0):
            raise FloatingPointError('Inadmissible field/spin Gaussian covariance')
        b = cross/lambda1 if lambda1 else 0.
        variance = self.r**2-b**2
        if variance < 0:
            raise FloatingPointError('Inadmissible field/spin Gaussian covariance')
        x = self.h-self.p.j0*m
        # Optimizer weights absorb these deterministic spin terms. Reconstruct
        # them from the stored physical logchi and accepted multipliers here.
        logweights = self.logchi.copy()
        logweights += np.array([1., -1.])[:, None]*(self.mu+self.ok*self.theta0*x)
        return self._gaussian_measure(m, q, logweights, b, variance)

    def _observables(self, measure: np.ndarray, m: float):
        """Contract the auxiliary measure into density and saddle moments.

        First integrate spin/field at fixed Gaussian nodes, THEN average their
        products with gw. In particular <s>**2 is formed before the outer
        average: replacing it by the square of the total magnetization is wrong.
        sr and hr have rho's shape and represent density-weighted conditional
        spin and centred-field moments used in Phi3; they are not velocities.
        rterm and tterm are the outer averages <X>**2 and <s><X>, X=H-J0*m.
        """
        rho_calc = np.einsum('ij,sijh->sh', self.gw, measure)
        spin_moment = self.dh*np.sum(measure[0]-measure[1], axis=-1)
        field_moment = self.dh*np.einsum('sijh,h->ij', measure, self.h-self.p.j0*m)
        sr = np.einsum('ij,sijh,ij->sh', self.gw, measure, spin_moment)
        hr = np.einsum('ij,sijh,ij->sh', self.gw, measure, field_moment)
        qnew = np.sum(self.gw*spin_moment**2)
        rterm = np.sum(self.gw*field_moment**2)
        tterm = np.sum(self.gw*spin_moment*field_moment)
        return rho_calc, sr, hr, qnew, rterm, tterm

    def saddle_residuals(self, rho, *, measure=None, m=None, q=None,
                         mu=None, theta0=None, theta1=None, Q0=None):
        """Independently evaluate every equation (6.15), plus normalization/PSD.

        Each residual is dimensionless: density, Theta0 and Theta1 are multiplied
        by j1. Density is an infinity norm, not an average that can hide bad cells.
        Expectations here are taken under the candidate measure, not substituted
        from the target density. This is separate from the optimization evaluation.
        """
        # Use explicit trial values when supplied; otherwise check stored state.
        m = self.dh*np.sum(rho[0]-rho[1]) if m is None else m
        q = self.q if q is None else q
        if not 0 <= q < 1:
            raise ValueError('Require 0<=q<1 for the real RS measure')
        mu = self.mu if mu is None else mu
        theta0 = self.theta0 if theta0 is None else theta0
        theta1 = self.theta1 if theta1 is None else theta1
        Q0 = self.r**2 if Q0 is None else Q0
        measure = self._measure(m, q) if measure is None else measure
        # Recompute the moments independently of the optimizer's residual vector.
        x = self.h-self.p.j0*m
        spin = np.array([1., -1.])[:, None, None, None]
        conditional_spin = self.dh*np.sum(spin*measure, axis=(0, 3))
        conditional_field = self.dh*np.sum(x*measure, axis=(0, 3))
        mean_spin = np.sum(self.gw*conditional_spin)
        mean_field = np.sum(self.gw*conditional_field)
        mean_square_field = self.dh*np.sum(self.gw[None, :, :, None]*measure*x**2)
        mean_spin_field = self.dh*np.sum(self.gw[None, :, :, None]*measure*spin*x)
        cross = np.sum(self.gw*conditional_spin*conditional_field)
        field_pair = np.sum(self.gw*conditional_field**2)
        # Evaluate the right-hand sides of the original multiplier equations.
        omq = 1-q
        mu_rhs = -self.p.j0*self.ok*m*theta0+self.p.j0*mean_field/(self.js*omq)
        theta0_rhs = (mean_spin_field-cross)/(2*self.js*omq**2)
        theta1_rhs = (mean_spin_field-2*self.js*(1-q*q)*theta0)/(2*self.js*omq**2)
        Q0_rhs = (q/omq**2-4*self.js*self.ok*theta0*(q*theta0+2*omq*theta1)
                  +(-2*q*mean_square_field+(1+q)*field_pair)/(self.js*omq**3))
        # Package signed equation errors and nonnegative norm/admissibility checks.
        density = np.einsum('ij,sijh->sh', self.gw, measure)
        return dict(density=self.p.j1*np.max(np.abs(density-rho)),
                    m=m-mean_spin, mu=mu-mu_rhs,
                    q=q-np.sum(self.gw*conditional_spin**2),
                    theta0=self.p.j1*(theta0-theta0_rhs),
                    theta1=self.p.j1*(theta1-theta1_rhs), Q0=Q0-Q0_rhs,
                    normalization=np.max(np.abs(self.dh*measure.sum(axis=(0, 3))-1)),
                    covariance=max(0., -Q0, self.js*(self.ok*theta1)**2-q*Q0/omq**2))

    def saddle(self, rho: np.ndarray, progress=None):
        """Solve the coupled RS equations with a PSD Gaussian covariance.

        Eliminate mu and Theta0 algebraically and absorb their deterministic
        exponent into logweights. Fix one logweight to remove the common-scale
        gauge. Minimize density and all original scalar equations together;
        independently check ALL equations before committing any model state.

        Solver coordinates are (active logweights except one fixed reference,
        q, cross-coordinate, nonnegative conditional variance). The cross-
        coordinate is b for k<1 and Theta1 itself for k=1. evaluate() constructs
        a candidate without mutating the accepted state; objective()/jacobian()
        fit it; checks() is the independent acceptance gate. The return value
        includes the accepted measure, transport moments and solver diagnostics.
        """
        # Validate the incoming density before constructing optimizer coordinates.
        if (rho.shape != (2, self.p.cells) or not np.all(np.isfinite(rho))
                or np.min(rho) < 0 or abs(self.dh*rho.sum()-1) > 1e-9):
            raise ValueError('Saddle requires a finite, nonnegative, normalized density')
        # Derive the current magnetization and flatten spin/field indexing.
        m = self.dh*np.sum(rho[0]-rho[1])
        if abs(m) >= 1:
            raise ValueError('The RS measure is singular at |m|=1 (q=1)')
        x = self.h-self.p.j0*m
        spin = np.repeat([1., -1.], self.p.cells)
        xx = np.tile(x, 2)
        support = np.flatnonzero(rho.ravel() > 0)
        # Exact zeros remain outside the fitted support. Adding the same
        # constant to all logweights leaves the normalized measure unchanged,
        # so fixing one coordinate removes a redundant optimizer direction.
        # Use the most populated cell as the fixed reference logweight.
        reference = np.argmax(rho.ravel()[support])
        support[[reference, -1]] = support[[-1, reference]]
        count = len(support)
        q_upper = np.nextafter(1., 0.)
        # Collect target moments and rebuild the previous solve's effective weights.
        mean_field = self.dh*np.dot(x, rho.sum(axis=0))
        spin_field = self.dh*np.dot(x, rho[0]-rho[1])
        logweights = self.logchi.ravel()+spin*(self.mu+self.ok*self.theta0*xx)
        if not np.all(np.isfinite(logweights[support])):
            raise FloatingPointError('Nonfinite log weight on positive-density support')
        # Translate the accepted multipliers into a continuation guess.
        q0 = float(self.q)
        lam0 = np.sqrt(q0)/(self.p.j1*(1-q0))
        b0 = self.ok*self.theta1/lam0 if lam0 else 0.
        # At k=1, covariance cross term (1-k^2)*Theta1 vanishes exactly.
        # Optimize Theta1 directly instead of recovering it by division by zero.
        fully_asymmetric = self.p.asymmetry == 1.
        variance0 = self.r**2-b0*b0
        # Rounding in reconstruction of an already PSD warm start is harmless;
        # optimizer trial covariances themselves are never clipped.
        if variance0 < -1e-12:
            raise FloatingPointError('Inadmissible warm-start Gaussian covariance')
        # Pack the free weights and three scalar coordinates into one fit vector.
        start = np.r_[logweights[support[:-1]]-logweights[support[-1]],
                      q0, self.theta1 if fully_asymmetric else b0, max(0., variance0)]
        cache = {}

        def evaluate(z):
            """Reconstruct trial multipliers, measure and ORIGINAL residuals.

            Cache repeated objective/Jacobian calls at the same coordinates.
            Copies are taken where later finite-difference evaluations would
            otherwise overwrite the cached candidate being differentiated.
            """
            # Reuse the previous trial when SciPy requests it again unchanged.
            if 'z' in cache and np.array_equal(z, cache['z']):
                return cache
            # Unpack trial coordinates, restoring the fixed reference and zeros.
            u = np.full(rho.size, -np.inf)
            u[support] = np.r_[z[:count-1], 0.]
            q, cross_coordinate, variance = z[-3:]
            b = 0. if fully_asymmetric else cross_coordinate
            omq = 1-q
            lam = np.sqrt(q)/(self.p.j1*omq)
            measure = self._gaussian_measure(m, q, u.reshape(rho.shape), b, variance)
            target, sr, hr, qt, rt, tt = self._observables(measure, m)
            # Eliminate Theta0 and mu using target-density moments and trial
            # contractions. Residuals still check the candidate measure's own
            # scalar equations, so finite density error cannot be hidden here.
            theta0 = (spin_field-tt)/(2*self.js*omq**2)
            theta1 = cross_coordinate if fully_asymmetric else lam*b/self.ok
            Q0 = b*b+variance
            mu = self.p.j0*mean_field/(self.js*omq)-self.ok*self.p.j0*m*theta0
            # Minimize the ORIGINAL equations, evaluated with candidate moments.
            # Substituting target moments is equivalent only at an exact density
            # match and can hide a much larger multiplier error at a bounded fit.
            fitted_field = self.dh*np.dot(x, target.sum(axis=0))
            fitted_square = self.dh*np.dot(x*x, target.sum(axis=0))
            fitted_spin_field = self.dh*np.dot(x, target[0]-target[1])
            Q0_target = (q/omq**2-4*self.js*self.ok*theta0*(q*theta0+2*omq*theta1)
                         +(-2*q*fitted_square+(1+q)*rt)/(self.js*omq**3))
            residual = np.r_[self.p.j1*(target-rho).ravel(),
                             m-self.dh*np.sum(target[0]-target[1]),
                             self.p.j0*(mean_field-fitted_field)/(self.js*omq), q-qt,
                             self.p.j1*(spin_field-fitted_spin_field)/(2*self.js*omq**2),
                             self.p.j1*(theta1-(fitted_spin_field-2*self.js*(1-q*q)*theta0)
                                        /(2*self.js*omq**2)), Q0-Q0_target]
            # Keep all derived trial arrays together for objective/Jacobian reuse.
            cache.clear()
            cache.update(z=z.copy(), u=u, q=q, b=b, variance=variance,
                         measure=measure, target=target, sr=sr, hr=hr, rt=rt, tt=tt,
                         theta0=theta0, theta1=theta1, mu=mu, Q0=Q0, residual=residual)
            return cache

        def checks(candidate):
            """Check a trial using explicit values, not accepted self attributes."""
            return self.saddle_residuals(
                rho, measure=candidate['measure'], m=m, q=candidate['q'],
                mu=candidate['mu'], theta0=candidate['theta0'],
                theta1=candidate['theta1'], Q0=candidate['Q0'])

        # Track work and the best checked trial without altering accepted state.
        evaluations = 0
        best_point = None
        best_error = np.inf
        svd_recovered = False
        class EquationsSatisfied(Exception):
            """Internal early exit carrying an independently verified iterate."""

        def objective(z):
            """Count coupled evaluations and stop on verified equation accuracy.

            EquationsSatisfied is a successful internal escape from SciPy, not
            a solver failure. best_point is retained for the narrow SVD recovery
            path; its independent maximum residual is distinct from the
            least-squares norm used to select initialization proposals below.
            """
            nonlocal evaluations, best_point, best_error
            candidate = evaluate(z)
            evaluations += 1
            equation_errors = checks(candidate)
            maximum_error = max(abs(v) for v in equation_errors.values())
            if all(np.isfinite(v) for v in equation_errors.values()) and maximum_error < best_error:
                best_point, best_error = z.copy(), maximum_error
            # Report trial progress, then escape early if every equation passes.
            if progress is not None:
                progress(evaluations, maximum_error)
            if (all(np.isfinite(v) for v in equation_errors.values())
                    and maximum_error <= self.p.saddle_tol):
                raise EquationsSatisfied(z.copy())
            return candidate['residual']

        def jacobian(z):
            # Exact logweight derivatives; only the three covariance variables
            # need finite differences. Avoid O(cells) Gaussian re-evaluations.
            candidate = evaluate(z).copy()
            q = candidate['q']; omq = 1-q
            measure = candidate['measure'].transpose(1, 2, 0, 3).reshape(-1, rho.size)
            # Each flattened row is a conditional measure at one node pair.
            # Differentiating a normalized exponential gives a centred moment,
            # which supplies the covariance terms in these logweight columns.
            weights = self.gw.ravel()
            s = self.dh*(measure@spin)
            f = self.dh*(measure@xx)
            ds = self.dh*measure*(spin-s[:, None])
            df = self.dh*measure*(xx-f[:, None])
            dqt = 2*(weights*s)@ds
            drt = 2*(weights*f)@df
            dtt = (weights*s)@df+(weights*f)@ds
            dt0 = -dtt/(2*self.js*omq**2)
            t0 = candidate['theta0']
            t1 = candidate['theta1']
            density_jac = np.diag(candidate['target'].ravel())-self.dh*(measure.T*weights)@measure
            dm = self.dh*(spin@density_jac)
            dfield = self.dh*(xx@density_jac)
            dsquare = self.dh*((xx*xx)@density_jac)
            dspinfield = self.dh*((spin*xx)@density_jac)
            dQ_residual = (8*self.js*self.ok*dt0*(q*t0+omq*t1)
                          +(2*q*dsquare-(1+q)*drt)/(self.js*omq**3))
            # Assemble derivatives in exactly the objective's row/column order.
            jac = np.zeros((rho.size+6, count+2))
            jac[:, :count-1] = np.vstack((self.p.j1*density_jac,
                                         -dm, -self.p.j0*dfield/(self.js*omq), -dqt,
                                         -self.p.j1*dspinfield/(2*self.js*omq**2),
                                         self.p.j1*(-dspinfield/(2*self.js*omq**2)
                                                    +(1-q*q)*dt0/omq**2),
                                         dQ_residual))[:, support[:-1]]
            # Probe the remaining scalar coordinates with small bounded changes.
            for column in range(count-1, count+2):
                step = np.sqrt(np.finfo(float).eps)*max(1., abs(z[column]))
                if column == count-1 and z[column]+step >= q_upper:
                    step = -step
                shifted = z.copy(); shifted[column] += step
                jac[:, column] = (evaluate(shifted)['residual']-candidate['residual'])/step
            return jac

        candidate = evaluate(start).copy()
        # A verified continuation guess needs no optimization. Initialization
        # proposals and Jacobian probes are not coupled objective evaluations;
        # saddle_maxiter caps the latter, matching SciPy's max_nfev convention.
        residuals = checks(candidate)
        if progress is not None:
            progress(0, max(abs(v) for v in residuals.values()))
        iterations = 0
        optimizer_status = 'warm start checked without optimization'
        if max(abs(v) for v in residuals.values()) > self.p.saddle_tol:
            # Refine the initial guess by log-domain density-matching
            # (iterative proportional fitting) steps at fixed covariance. Tiny
            # predicted probabilities otherwise have effectively zero Jacobian
            # columns, even when their positive target densities require change.
            # Use the kernel without subtracting huge logweights from one another.
            # Refine only the effective logweights at FIXED warm-start
            # covariance. A single matching update can leave the coupled fit
            # in a poorly conditioned tail-weight configuration at low T.
            best_start = start.copy()
            best_merit = np.linalg.norm(candidate['residual'])
            for matching_step in range(4):
                u = np.full(rho.size, -np.inf)
                u[support] = np.r_[start[:count-1], 0.]
                log_density = self._gaussian_log_density(
                    m, start[-3], u.reshape(rho.shape), candidate['b'], start[-1]).ravel()
                # Algebraically u + log(rho / fitted_density), evaluated without
                # exponentiation or subtracting two large logweight terms.
                matched = u[support]+np.log(rho.ravel()[support])-log_density[support]
                start[:count-1] = matched[:-1]-matched[-1]
                guess = evaluate(start).copy()
                merit = np.linalg.norm(guess['residual'])
                if np.isfinite(merit) and merit < best_merit:
                    best_start, best_merit = start.copy(), merit
            # Matching is only an initializer: never discard a continuation
            # guess for a proposal with a worse least-squares objective norm.
            start = best_start
            # Bound q and the variance, leaving weights and cross-coordinate free.
            lower = np.r_[np.full(count-1, -np.inf), 0., -np.inf, 0.]
            upper = np.r_[np.full(count-1, np.inf), q_upper, np.inf, np.inf]
            # Run the coupled fit; handle its checked-success exit and narrow
            # SVD-recovery path separately from ordinary optimizer termination.
            try:
                fit_options = dict(jac=jacobian, bounds=(lower, upper),
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12)
                try:
                    fit = least_squares(objective, start, max_nfev=self.p.saddle_maxiter,
                                        **fit_options)
                except np.linalg.LinAlgError as error:
                    # A finite, ill-conditioned Jacobian can defeat LAPACK's
                    # default SVD. Keep the same TRF solver and equations, using
                    # one uniform coordinate rescaling, not a different optimizer.
                    if 'SVD did not converge' not in str(error):
                        raise
                    remaining = self.p.saddle_maxiter-evaluations
                    if remaining < 1 or best_point is None:
                        raise RuntimeError('SVD failed and no saddle evaluation budget remains') from error
                    svd_recovered = True
                    fit = least_squares(objective, best_point, x_scale=.5,
                                        max_nfev=remaining, **fit_options)
                # SciPy stopped normally; its termination message is diagnostic
                # information, not permission to skip the final equation checks.
                result = fit.x
                iterations = evaluations
                optimizer_status = getattr(fit, 'message', 'optimizer did not supply a termination message')
            except EquationsSatisfied as convergence:
                result = convergence.args[0]
                iterations = evaluations
                optimizer_status = 'all original equations satisfy saddle_tol'
            candidate = evaluate(result).copy()
            residuals = checks(candidate)
        # Reject the final trial with a named worst equation if any check fails.
        worst = max(residuals, key=lambda name: abs(residuals[name]))
        if not all(np.isfinite(v) for v in residuals.values()) or abs(residuals[worst]) > self.p.saddle_tol:
            details = ', '.join(f'{name}={value:.6g}' for name, value in residuals.items())
            raise RuntimeError(f'RS saddle did not satisfy all equations after {iterations} '
                               f'evaluations; worst={worst}; {details}; '
                               f'conditional_variance={candidate["variance"]:.6g}; '
                               f'optimizer={optimizer_status}')
        # Commit only a verified candidate. Failure leaves the warm start intact.
        self.q = candidate['q']; self.mu = candidate['mu']
        self.theta0 = candidate['theta0']; self.theta1 = candidate['theta1']
        self.r = np.sqrt(candidate['Q0'])
        logchi = candidate['u']-spin*(self.mu+self.ok*self.theta0*xx)
        logchi -= np.max(logchi)
        self.logchi = logchi.reshape(rho.shape)
        self.rterm = candidate['rt']; self.tterm = candidate['tt']
        self.last_saddle_residuals = residuals
        # Return the accepted closure and diagnostics needed by the time step.
        return dict(q=self.q, m=m, measure=candidate['measure'], sr=candidate['sr'],
                    hr=candidate['hr'], residual=max(abs(v) for v in residuals.values()),
                    residuals=residuals, objective_residual=candidate['residual'].copy(),
                    iterations=iterations, svd_recovered=svd_recovered)

    def coefficients(self,rho,sol):
        """Evaluate thesis (6.17) from rho and its accepted instantaneous saddle.

        Returns (Phi1, Phi2[spin], Phi3[spin,field], Phi4, tanh(beta*h)).
        Moment shorthand matches the README coefficient definitions:
        tp+tm=U, tp-tm=A, ts=B, ht=C, h_t=D; sol['sr']=S, sol['hr']=K.
        l0 and l1 are the thesis's lambda0**2 and lambda1**2 coefficients;
        l0 can be negative and must not be treated as a variance.
        Phi4 is field diffusion, not noise on the macroscopic density trajectory.
        """
        q,m = sol['q'],sol['m']
        x = self.h-self.p.j0*m
        th = np.tanh(self.h/self.p.temperature) if self.p.temperature else np.sign(self.h)
        tp = self.dh*np.dot(th,rho[0]); tm=self.dh*np.dot(th,rho[1])
        ts = self.dh*np.dot(th,sol['sr'].sum(axis=0))
        ht = self.dh*np.dot(x*th,rho.sum(axis=0))
        h_t = self.dh*np.dot(th,sol['hr'].sum(axis=0))
        l0=(1-2*q)/(self.js*(1-q)**2)
        l1=q/(self.js*(1-q)**2)
        phi1=1-self.js*l0*(tp-tm)-self.js*l1*ts
        base=self.p.j0*m-self.p.j0*(tp+tm)
        correction=self.js*self.ko*(l0*ht+l1*h_t)
        theta_c=2*self.js*self.ko*((self.theta0+self.theta1)*(tp-tm)-self.theta1*ts)
        phi2=np.array([base-correction+theta_c,base+correction-theta_c])
        phi3=(self.js*self.ko*sol['sr']*((l0+2*l1)*h_t-l1*ht)
              +self.js*sol['hr']*((l0+2*l1)*ts-l1*(tp-tm))
              -2*self.js*self.ko*sol['sr']*((self.theta0-self.theta1)*ts+self.theta1*(tp-tm)))
        # The hr term above has no ko factor: k=1 does not in general make
        # Phi3 zero, even though reciprocal-coupling terms disappear there.
        # Phi3 is a fitted-density-weighted conditional transport moment.
        # Divide by that SAME density before applying it to the evolving rho.
        # Absolute saddle tolerances do not control relative tail errors: mixing
        # fitted moments with rho in the velocity can otherwise create enormous
        # artificial speeds in tiny-probability cells. At an exact saddle this
        # is the original flux, and the independent equation checks are unchanged.
        fitted_density = np.einsum('ij,sijh->sh', self.gw, sol['measure'])
        conditional_transport = np.divide(phi3, fitted_density,
                                          out=np.zeros_like(phi3), where=fitted_density > 0)
        phi3 = rho*conditional_transport
        phi4=self.js*(1-tp+tm)
        # Nonnegative for a physical density. Permit only roundoff below zero;
        # materially negative diffusion indicates a broken numerical state.
        if phi4 < -1e-12:
            raise FloatingPointError('Negative diffusion coefficient')
        return phi1,phi2,phi3,max(phi4,0.),th

    def _face_transport_rates(self, velocity, diffusion):
        """Nonnegative Scharfetter-Gummel rates: flux = a*rho_left-b*rho_right.

        Exponential fitting solves the constant-coefficient drift/diffusion
        problem on each face interval. Evaluate the Bernoulli function using
        nonpositive exponentials, including the pure-advection limit D=0.

        For signed P=v*dh/D, a=(D/dh)*B(-P), b=(D/dh)*B(P),
        B(P)=P/expm1(P). The common part below is (D/dh)*B(abs(P));
        adding the appropriate signed-advection part gives a,b without forming
        exp(large positive P). These face coefficients have units field/time;
        division by dh gives the outgoing probability rates used by advance().
        """
        speed = np.abs(velocity)
        scale = diffusion/self.dh
        if scale == 0:
            common = np.zeros_like(speed)
        else:
            with np.errstate(over='ignore'):
                peclet = speed/scale
            common = np.empty_like(speed)
            small = peclet < 1e-4
            # Taylor series resolves 0/0 at zero drift; the other branch uses
            # exp(-P) and expm1(-P) to remain stable for strong drift/small D.
            p = peclet[small]
            common[small] = scale*(1-p/2+p*p/12-p**4/720)
            p = peclet[~small]
            common[~small] = speed[~small]*np.exp(-p)/(-np.expm1(-p))
        return common+np.maximum(velocity, 0.), common+np.maximum(-velocity, 0.)

    def advance(self,rho,coeff,dt):
        """Conservative exponential-fit transport followed by exact spin flips.

        Phi3 uses conditional moments normalized by their fitted saddle density.
        The effective advective flux Phi3/rho retains the existing near-zero
        density floor, so this still differs from the literal continuum equation.

        Freeze coefficients for one explicit field-transport step, then solve
        spin flips exactly at each field cell (first-order operator splitting).
        Return BOTH the updated density and the accepted step, which can be
        smaller than the caller's requested dt. Exact spin integration does not
        make the coupled transport/saddle evolution exact in time.
        """
        phi1,phi2,phi3,phi4,th=coeff
        x=self.h-self.p.j0*self.dh*(rho[0]-rho[1]).sum()
        # Equation (6.16) writes drift as +d_h[rho*A+Phi3]; in the flux
        # convention d_t rho=-d_h J this means velocity=-(A+Phi3/rho).
        adv=-(phi1*x[None,:]+phi2[:,None]+phi3/np.maximum(rho,1e-12))
        v=.5*(adv[:,:-1]+adv[:,1:])
        right, left = self._face_transport_rates(v, phi4)
        # Zero boundary flux. Limit the sum of outgoing rates at each cell,
        # rather than separate drift/diffusion bounds, to ensure positivity.
        outgoing = np.zeros_like(rho)
        outgoing[:, :-1] += right
        outgoing[:, 1:] += left
        max_rate = np.max(outgoing)/self.dh
        dt=min(dt, self.p.dt_max, .8/max_rate if max_rate > 0 else np.inf)
        fraction = dt/self.dh
        # Each cell retains its mass minus outgoing transfers and receives the
        # two neighbouring incoming transfers. There are no external faces.
        # This form conserves mass and avoids subtracting almost equal fluxes
        # in cells with extremely small probabilities. No renormalization follows.
        updated=rho*(1-fraction*outgoing)
        updated[:, :-1] += fraction*left*rho[:, 1:]
        updated[:, 1:] += fraction*right*rho[:, :-1]
        if np.min(updated) < -1e-11:
            raise FloatingPointError('Spatial flux step lost positivity')
        # Exact solution of two-state heat-bath spin-flip kinetics at each h.
        # rho_plus relaxes at rate 1 toward (1+tanh(beta*h))*total/2;
        # total=rho_plus+rho_minus is unchanged by a spin-label flip.
        total=updated.sum(axis=0)
        equilibrium_plus=.5*(1+th)*total
        updated[0]=equilibrium_plus+(updated[0]-equilibrium_plus)*np.exp(-dt)
        updated[1]=total-updated[0]
        return updated,dt

    def energy(self, rho):
        """Energy observable -1/2 integral h*(rho_plus-rho_minus) dh, (6.2).

        h is the FULL local field: do not subtract J0*m or add a separate mean-
        coupling term. For k=0 this is Hamiltonian energy per spin. At k>0 it
        remains a spin-field observable, not a Hamiltonian for directed dynamics.
        """
        return -.5*self.dh*np.dot(self.h, rho[0]-rho[1])

    def run(self,t_end=.02,magnetization=.2,progress_interval=None):
        """Start a fresh density and alternate saddle solves with accepted steps.

        Return (trajectory, final_density). Trajectory columns are time, m, q,
        mass, maximum saddle residual. Time/m/mass and energy_history describe
        the UPDATED density; q/residual describe the PRE-UPDATE saddle, whose
        actual times are recorded separately in saddle_times. There is no extra
        final-time saddle solve. A zero-duration run returns no trajectory rows.
        progress_interval is wall-clock reporting cadence, not simulated time.
        """
        # Check run duration and optional wall-clock reporting cadence.
        if not math.isfinite(t_end) or t_end < 0:
            raise ValueError('t_end must be finite and nonnegative')
        if progress_interval is not None and (not math.isfinite(progress_interval) or progress_interval <= 0):
            raise ValueError('progress_interval must be finite and positive')
        # Start a fresh trajectory and empty all per-step diagnostic histories.
        started = time.monotonic()
        last_report = started
        rho=self.initial(magnetization)
        records=[]; t=0.; n=0
        self.saddle_history = []
        self.saddle_times = []
        self.energy_history = []
        self.svd_recoveries = 0
        self.saddle_iterations = []

        def report(stage, force=False):
            nonlocal last_report
            # Direct library calls are quiet unless reporting was requested.
            if progress_interval is None:
                return
            # Throttle messages by real elapsed time, not simulated step count.
            now = time.monotonic()
            if force or now - last_report >= progress_interval:
                percent = min(100., 100*t/t_end) if t_end > 0 else 100.
                print(f'{stage}: t={t:.6g}/{t_end:.6g} ({percent:.1f}%), '
                      f'steps={n}, elapsed={now-started:.1f}s', flush=True)
                last_report = now

        def saddle_progress(evaluation, residual):
            # Reuse the same throttle during potentially expensive saddle solves.
            report(f'Solving saddle for step {n+1}, evaluation={evaluation}, residual={residual:.3g}')

        # Announce the start even when the run is too short for periodic messages.
        report('Starting simulation', force=True)
        while t < t_end-1e-14:
            # A failed saddle cannot supply transport coefficients. Attach only
            # completed steps to supported saddle exceptions so main() can save
            # a labelled partial archive before propagating the failure.
            try:
                sol=self.saddle(rho, progress=saddle_progress if progress_interval is not None else None)
            except (FloatingPointError, RuntimeError) as error:
                failure = type(error)(f'Saddle failed at t={t:.9g}, step={n+1} '
                                      f'({n} completed steps): {error}')
                # Only completed steps and their last valid density are retained.
                failure.partial_trajectory = np.asarray(records).reshape(-1, 5)
                failure.partial_rho = rho.copy()
                failure.completed_time = t
                raise failure from error
            self.svd_recoveries += int(sol['svd_recovered'])
            self.saddle_iterations.append(sol['iterations'])
            coeff=self.coefficients(rho,sol)
            # Record the instantaneous closure before advancing rho; preserve
            # that time distinction in archives rather than shifting q forward.
            self.saddle_times.append(t)
            self.saddle_history.append(sol['residuals'].copy())
            rho,dt=self.advance(rho,coeff,t_end-t)
            t+=dt; n+=1
            # Check the completed step before appending its observable history.
            mass=self.dh*rho.sum()
            if abs(mass-1)>1e-9 or np.min(rho)<-1e-10:
                raise FloatingPointError(f'Conservation/positivity failure at t={t}: mass={mass}')
            records.append([t,self.dh*(rho[0]-rho[1]).sum(),sol['q'],mass,sol['residual']])
            self.energy_history.append(self.energy(rho))
            report('Running')
        # Return completed rows and the final density; histories stay on the model.
        report('Simulation complete', force=True)
        return np.asarray(records),rho


def read_config(path):
    """Read scalar name=value settings and canonicalize aliases before checking.

    T maps to temperature; fd_* names map to this solver's unprefixed controls.
    Reject duplicate aliases and unknown settings rather than silently dropping
    them. Return supplied settings after alias/output resolution; main() merges
    them with run defaults.
    Output paths remain relative to the working directory, not the config file.
    """
    # Use defaults to identify allowed names and the expected type of each value.
    defaults = asdict(Parameters()) | dict(t_end=.01, m0=.2, output='trajectory.npz',
                                         output_dir='.', output_prefix='trajectory')
    settings = {}
    # Read one assignment per line, ignoring comments and blank lines.
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        # Split the assignment and translate accepted aliases to internal names.
        name, separator, value = line.partition('=')
        name, value = name.strip(), value.strip()
        name = 'temperature' if name == 'T' else name
        if name.startswith('fd_') and name[3:] in asdict(Parameters()):
            name = name[3:]
        # Include the source line in errors for malformed or repeated settings.
        location = f'{path}:{number}'
        if not separator or name not in defaults:
            raise ValueError(f'{location}: expected a known parameter name = value')
        if name in settings:
            raise ValueError(f'{location}: duplicate parameter {name}')
        # Convert the text to the setting's declared type, then check its value.
        try:
            parsed = type(defaults[name])(value)
        except ValueError:
            raise ValueError(f'{location}: invalid value for {name}: {value!r}') from None
        if name == 'output' and not parsed:
            raise ValueError(f'{location}: output must not be empty')
        if name not in ('output', 'output_dir', 'output_prefix') and not math.isfinite(parsed):
            raise ValueError(f'{location}: {name} must be finite')
        settings[name] = parsed
    # Convert output_dir/output_prefix, if supplied, into the common output path.
    from parameter_files import resolve_output_settings
    return resolve_output_settings(settings, defaults['output'])


def reserve_output(output, started):
    """Create a timestamped output exclusively, including on timestamp collisions."""
    # Build a timestamped base name, adding the extension exactly once.
    path = Path(output)
    stem = path.name[:-4] if path.name.endswith('.npz') else path.name
    base = f'{stem}_{started:%Y-%m-%d_%H%M%S_%f}'
    counter = 0
    # Reserve the file atomically; try a numbered suffix if the name exists.
    while True:
        suffix = f'_{counter}' if counter else ''
        candidate = path.with_name(f'{base}{suffix}.npz')
        try:
            return candidate, candidate.open('xb')
        except FileExistsError:
            counter += 1


def main(argv=None):
    """Resolve inputs, run once, save provenance/diagnostics, then report failure.

    Precedence is CLI run defaults < parameter file < explicit CLI options.
    Library Parameters defaults and CLI run defaults are intentionally separate.
    Only supported saddle failures with attached completed-step state are saved
    as partial runs; arbitrary exceptions or process termination are not restart
    checkpoints. A saved failed run still exits with an error after writing.
    """
    # Capture one local start timestamp for output naming and provenance.
    started = datetime.now().astimezone()
    # Declare accepted command-line arguments and let argparse read/validate
    # their syntax. SUPPRESS keeps omitted options out of the override dictionary.
    parser=argparse.ArgumentParser(description='Experimental Appendix D spin glass solver')
    parser.add_argument('parameter_file', nargs='?', help='text file containing name = value settings')
    parser.add_argument('--config',help='text file containing name = value settings')
    parser.add_argument('--t-end',type=float,default=argparse.SUPPRESS)
    parser.add_argument('--m0',type=float,default=argparse.SUPPRESS)
    parser.add_argument('--cells',type=int,default=argparse.SUPPRESS)
    parser.add_argument('--quadrature',type=int,default=argparse.SUPPRESS)
    parser.add_argument('--temperature',type=float,default=argparse.SUPPRESS)
    parser.add_argument('--j0',type=float,default=argparse.SUPPRESS)
    parser.add_argument('--output',default=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    # Accept either spelling of the config argument, but never two config files.
    if args.parameter_file and args.config:
        parser.error('Specify either a positional parameter file or --config, not both')
    config = args.parameter_file or args.config
    # Start from CLI run defaults, including settings outside the model dataclass.
    settings = asdict(Parameters()) | dict(cells=32, gaussian_order=8, t_end=.01, m0=.2,
                                          output='trajectory.npz')
    try:
        # Apply file settings first so explicit command-line values can win later.
        if config:
            settings.update(read_config(config))
        # Collect only explicitly supplied options and translate --quadrature.
        overrides = {key: value for key, value in vars(args).items()
                     if key not in ('config', 'parameter_file')}
        if 'quadrature' in overrides:
            overrides['gaussian_order'] = overrides.pop('quadrature')
        settings.update(overrides)
        # Check the merged values before allocating grids or starting a run.
        if not all(math.isfinite(value) for key, value in settings.items() if key != 'output'):
            raise ValueError('All parameters must be finite')
        if settings['t_end'] < 0 or abs(settings['m0']) > 1:
            raise ValueError('Require t_end>=0 and |m0|<=1')
        if settings['dt_max'] <= 0 or settings['saddle_tol'] <= 0:
            raise ValueError('Require dt_max>0 and saddle_tol>0')
        if settings['saddle_maxiter'] < 1 or not 0 < settings['saddle_damping'] <= 1:
            raise ValueError('Require saddle_maxiter>=1 and 0<saddle_damping<=1')
        # Pass only model/numerical fields to Parameters; keep run/output options
        # in settings for the driver and saved metadata.
        parameters = Parameters(**{key: settings[key] for key in asdict(Parameters())})
    except (OSError, ValueError) as error:
        # Present input problems as concise CLI errors rather than tracebacks.
        parser.error(str(error))
    # Construct the model and run it with periodic progress messages enabled.
    model=SpinGlass(parameters)
    failure = None
    try:
        trajectory,rho=model.run(settings['t_end'],settings['m0'],progress_interval=10.)
    except (RuntimeError, FloatingPointError) as error:
        # Save recoverable completed-step data, but do not hide unrelated errors.
        if not hasattr(error, 'partial_trajectory'):
            raise
        failure = error
        trajectory, rho = error.partial_trajectory, error.partial_rho
    # Retain the original metadata keys alongside the complete effective settings.
    metadata = settings | dict(quadrature=parameters.gaussian_order,
                               config=config,
                               run_status='failed' if failure else 'complete',
                               completed_time=(failure.completed_time if failure else
                                               float(trajectory[-1, 0]) if len(trajectory) else 0.),
                               failure_message=str(failure) if failure else None)
    # Reserve a unique destination and record how transport/solver recovery ran.
    output_path, output_file = reserve_output(settings['output'], started)
    metadata.update(output_file=str(output_path), run_started=started.isoformat(),
                    transport_normalization='fitted_saddle_density', transport_density_floor=1e-12,
                    svd_recoveries=model.svd_recoveries,
                    max_saddle_evaluations=max(model.saddle_iterations,default=0))
    # Keep equation diagnostics as numeric arrays with a fixed column-name list.
    # JSON strings hold settings and algorithm identifiers without requiring
    # pickled Python objects when loading the NPZ with allow_pickle=False.
    residual_names = ('density', 'm', 'mu', 'q', 'theta0', 'theta1', 'Q0',
                      'normalization', 'covariance')
    residual_history = np.array([[row[name] for name in residual_names]
                                 for row in model.saddle_history]).reshape(-1, len(residual_names))
    # Identify the numerical methods so old and new archives can be distinguished.
    solver_info = dict(method='bounded_least_squares_RS_v3',
                       residual='maximum absolute dimensionless equation residual',
                       weight_representation='persistent_logchi',
                       initial_guess='log_domain_density_matching', density_matching_steps=4,
                       initial_guess_selection='lowest_objective_norm_including_warm_start',
                       stopping_rule='all_equations_within_saddle_tol',
                       svd_recovery='one_half_scale_TRF_retry_within_evaluation_budget',
                       measure_evaluation='factorized_field_spin_normalization_v1',
                       spatial_flux='scharfetter_gummel_v1',
                       transport_timestep='0.8_over_max_outgoing_rate',
                       inactive_parameters=['saddle_damping'])
    # Write all results to the reserved NPZ and close it even if writing fails.
    with output_file:
        np.savez(output_file,trajectory=trajectory,rho=rho,h=model.h,parameters=json.dumps(metadata),
                 saddle_residuals=residual_history, saddle_residual_names=np.array(residual_names),
                 saddle_times=np.array(model.saddle_times), solver=json.dumps(solver_info),
                 energy=np.array(model.energy_history),
                 saddle_iterations=np.array(model.saddle_iterations,dtype=int))
    # Print the actual saved path, including the timestamp/collision suffix.
    print('Saved',output_path,'steps=',len(trajectory),'final=',trajectory[-1] if len(trajectory) else 'initial', flush=True)
    if failure is not None:
        # A partial archive is evidence of a failed run, not a successful exit.
        raise failure

# Run the CLI only when invoked as a script; imports expose the library API.
if __name__=='__main__':
    main()
