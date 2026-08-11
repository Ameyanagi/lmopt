# Migrating to lmopt 0.2

Version 0.2 fixes solver correctness issues and makes several intentional API
changes.

## Errors

`lmopt::Result<T>` now uses the matchable `lmopt::Error` enum instead of
`anyhow::Error`. Applications that want contextual error chains can map the
library error into `anyhow` at their boundary.

Analytical Jacobians that can fail should override `try_jacobian`. Existing
infallible `jacobian` implementations continue to work.

## Jacobian selection

The default is `JacobianMethod::Auto`: it uses an analytical Jacobian when the
problem supplies one and central finite differences otherwise. Select
`UserProvided` when a missing analytical Jacobian should be an error.

`JacobianMethod::AutoDiff` now returns `Error::AutoDiffUnavailable`; it no
longer reports finite differences as autodiff. The validated Enzyme experiment
is intentionally outside the stable API.

## Reports

`MinimizationReport` adds:

- `residual_evaluations` and `jacobian_evaluations`;
- `accepted_steps` and `rejected_steps`;
- `final_lambda`; and
- `svd_fallbacks`.

Code constructing reports directly must initialize these fields. Most callers
only consume reports and require no change.

## Features and toolchain

The `ndarray` and `nalgebra` conversions now require their corresponding crate
features. The core crate has neither enabled by default.

The minimum supported Rust version is 1.85. The main optimizer and every stable
feature remain compatible with stable Rust; only `experiments/enzyme` uses a
pinned nightly toolchain.
