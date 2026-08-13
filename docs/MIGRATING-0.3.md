# Migrating to lmopt 0.3

Version 0.3 makes convergence scale-independent and adds a simpler API for the
common `f64` case. It also contains intentional breaking changes so incorrect
or incomplete optimization is harder to accept accidentally.

## Simple closure API

Most callers no longer need to define a problem type or manipulate faer
matrices:

```rust
let report = lmopt::least_squares(&[0.0, 0.0], |parameters| {
    let [slope, intercept] = parameters else { return Vec::new() };
    [
        slope * 0.0 + intercept - 1.0,
        slope * 1.0 + intercept - 3.0,
        slope * 2.0 + intercept - 5.0,
    ]
    .to_vec()
})?;
# Ok::<(), lmopt::Error>(())
```

Use `try_least_squares` for fallible callbacks. Its error can be any type that
implements `Display`. The methods `minimize_fn`, `optimize_fn`,
`try_minimize_fn`, and `try_optimize_fn` provide the same interface on a
configured `LevenbergMarquardt` value.

The advanced methods now accept any faer `AsMatRef`, so a `Col` can be passed
without converting it to a one-column `Mat`. Call `report.parameters()` to
borrow the result as a plain slice.

## Checked non-convergence

`minimize` now returns `Error::NoConvergence` when it reaches the iteration
limit or cannot make floating-point progress. Use `optimize` when you need the
best partial `MinimizationReport` even if the solver did not converge. The
closure API follows the same `minimize_fn`/`optimize_fn` distinction.

`MinimizationReport` is now non-exhaustive. Code that destructures it must use
`..`. Prefer its `parameters()` and `converged()` convenience methods for the
common cases.

## Convergence settings

`with_epsilon_1` and `with_epsilon_2` remain as deprecated aliases. New code
should use:

- `with_ftol` for relative objective reduction;
- `with_xtol` for the relative scaled step; and
- `with_gtol` for the scaled gradient.

Defaults are selected for the scalar precision, and the default numerical
finite-difference step is likewise adapted for `f32` or `f64`. Damping and
convergence checks are invariant to uniform residual scaling.

## Removed API

The unused `InvalidJacobian`, `InvalidResiduals`, and `Other` termination
variants were removed; those conditions are returned as typed errors. The
empty `minpack-compat` feature was also removed.
