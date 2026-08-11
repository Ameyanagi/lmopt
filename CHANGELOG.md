# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An isolated CubeCL/WGPU feasibility benchmark for GPU residual and
  analytical-Jacobian evaluation.

### Removed

- The experimental Enzyme crate and the unavailable `AutoDiff` API surface.

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

[Unreleased]: https://github.com/Ameyanagi/lmopt/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Ameyanagi/lmopt/compare/04003d1...v0.2.0
