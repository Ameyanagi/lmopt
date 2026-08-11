# lmopt development guide

## Project focus

`lmopt` implements dense Levenberg-Marquardt nonlinear least squares using
`faer`. The core library targets stable Rust. Near-term work should prioritize:

1. Numerical correctness and parity with established LM implementations.
2. Useful diagnostics and a predictable public API.
3. Measured performance improvements.
4. Parameter constraints and uncertainty analysis after the core solver is
   well validated.

See `FAER.md` for the pinned faer API used by this repository.

## Current Jacobian strategies

- `JacobianMethod::Auto` is the default. It uses an analytical Jacobian when
  the problem supplies one and central finite differences otherwise.
- `UserProvided`, `NumericalForward`, `NumericalCentral`, and
  `NumericalBackward` force a particular implemented strategy.
- `AutoDiff` is reserved for a future Enzyme backend and currently returns
  `Error::AutoDiffUnavailable`. Never silently substitute finite differences
  while reporting autodiff.

## Enzyme roadmap

Rust's `std::autodiff` integration is experimental and nightly-only. A real
implementation must not differentiate through the current `dyn EraseTypes`
path. Introduce a separate monomorphized, allocation-free residual interface
using slices or faer views, then benchmark forward and reverse modes against
analytical and numerical Jacobians. CI for that feature will need the rustup
Enzyme component, release mode, and fat LTO. Do not add the old `enzyme` crate;
it is an installation helper rather than the compiler backend.

## Build and validation

```text
cargo fmt --all -- --check
cargo test --all-targets
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

Run examples in release mode when inspecting solver behavior:

```text
cargo run --release --example linear_fitting
cargo run --release --example gaussian_fitting
cargo run --release --example jacobian_methods
```

## Development rules

- Write a failing regression test before fixing numerical behavior.
- Public integration tests must be top-level Cargo test targets. Tests of
  private internals belong in `#[cfg(test)]` modules beside the source.
- Use faer's decompositions and reductions instead of handwritten numerical
  kernels when faer provides the operation.
- Check dimensions and finite values at every user-function boundary.
- Rejected trust-region steps must never count as convergence.
- Reuse residuals and normal-equation products while only the damping value
  changes.
- Keep optional integrations behind features so the core dependency graph
  stays small.
- Add benchmarks before claiming a change is faster. Include warm-up,
  evaluation counts, and representative small and large problems.

## Public API conventions

- `LeastSquaresProblem::residuals` returns an owned residual column vector.
- `jacobian` is the convenient infallible analytical hook.
- Override `try_jacobian` when analytical Jacobian evaluation can fail.
- Use the library's typed `Error` and `Result`; application-level callers can
  add `anyhow` context themselves.
- `MinimizationReport` must describe the method actually used and include
  residual/Jacobian evaluation counts plus accepted/rejected step counts.

## Planned phase 2

After reference comparisons and benchmark coverage are in place, add:

- Named parameters, bounds, fixed parameters, and linked parameters.
- Covariance, confidence intervals, correlations, and goodness-of-fit data.
- Robust losses, multiple datasets, and constrained fitting.

These features should build on the existing typed errors and report structure
without weakening the correctness checks in the core solver.
