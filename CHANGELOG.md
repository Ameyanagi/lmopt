# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-16

### Added

- A closure-first API: `least_squares`, `try_least_squares`, and matching
  optimizer methods accept plain parameter slices and residual vectors.
- A scaled-gradient convergence criterion and precision-aware defaults for
  `f32` and `f64`.
- An external-consumer compile check for the README quick start and a
  retry-heavy optimizer benchmark.

### Changed

- Made trust-region damping and convergence checks invariant to uniform
  residual scaling, including very large and very small scales.
- Made `minimize` return `Error::NoConvergence` for unsuccessful termination;
  `optimize` retains access to partial reports.
- Accepted faer column vectors and other `AsMatRef` inputs directly, and added
  `MinimizationReport::parameters` for allocation-free slice output.
- Reused the augmented linear-system workspace across rejected trust-region
  trials.
- Replaced ambiguous epsilon settings with `ftol`, `xtol`, and `gtol` while
  retaining deprecated aliases.
- Made `MinimizationReport` non-exhaustive so diagnostics can evolve without
  breaking downstream destructuring.
- Enforced safe Rust in the library. Library failures use `thiserror`; examples
  and tests use `anyhow` at application boundaries instead of unchecked value
  extraction.

### Removed

- The experimental Enzyme crate and the unavailable `AutoDiff` API surface.
- The empty `minpack-compat` feature.

## [0.2.0] - 2026-08-11

### Added

- Automatic analytical-or-central Jacobian selection with
  `JacobianMethod::Auto`.
- Residual/Jacobian evaluation counts, accepted/rejected step counts, final
  damping, and SVD fallback counts in `MinimizationReport`.
- Fallible `LeastSquaresProblem::try_jacobian` support.
- Criterion benchmarks for Jacobian and optimizer performance.
- Generated and adversarial regression coverage for exact, noisy,
  underdetermined, ill-conditioned, and one-sided-domain problems.
- CI gates for formatting, Clippy, tests, documentation, feature combinations,
  Rust 1.85, and benchmark compilation.

### Changed

- Fixed accepted and rejected steps being incorrectly reported as convergence.
- Replaced normal-equation Cholesky solving with direct augmented
  column-pivoted QR and a tolerance-aware truncated-SVD fallback.
- Reused baseline residuals and linearizations across numerical Jacobian and
  rejected damping calculations.
- Made ndarray and nalgebra interoperability optional.
- Changed the crate result alias to the structured `lmopt::Error` type.
- Updated the minimum supported Rust version to 1.85.

### Removed

- Unsafe type-name-based matrix conversion dispatch.
- Unused unconditional dependencies and inaccessible duplicate test modules.

[Unreleased]: https://github.com/Ameyanagi/lmopt/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Ameyanagi/lmopt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ameyanagi/lmopt/compare/04003d1...v0.2.0
