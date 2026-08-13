# lmopt - Levenberg-Marquardt Optimization with faer

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

A robust Rust implementation of the Levenberg-Marquardt algorithm for dense
nonlinear least-squares optimization, using
[faer](https://github.com/sarah-ek/faer-rs) for linear algebra.

## Features

- **Powerful Optimizer**: Robust implementation of the Levenberg-Marquardt algorithm with trust region strategy
- **Measured Performance**: Criterion baselines cover Jacobian construction and end-to-end solves
- **Robust Linear Solves**: Damped least squares uses column-pivoted QR with a truncated-SVD fallback for rank-deficient systems
- **Multiple Jacobian Methods**:
  - Automatic selection of the best available implemented method
  - User-provided analytical Jacobian
  - Numerical differentiation (central, forward, or backward differences)
- **Matrix Interoperability**: Optional conversion support for `ndarray` and `nalgebra`
- **Comprehensive API**: Fluent interface for configurability and ease of use
- **Error Handling**: A structured, matchable error type built with `thiserror`

## Quick Start

```rust
use anyhow::{Context, Result};
use lmopt::least_squares;

fn main() -> Result<()> {
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [3.1, 5.2, 6.9, 9.1, 11.0, 13.1];

    // Fit y = slope * x + intercept, starting from [1, 1].
    let fit = least_squares(&[1.0, 1.0], |parameters| {
        let [slope, intercept] = parameters else { return Vec::new() };
        xs.iter()
            .zip(ys)
            .map(|(x, y)| slope * x + intercept - y)
            .collect()
    })
    .context("failed to fit the line")?;

    let parameters = fit.parameters();
    println!("Fitted line: y = {:.4}*x + {:.4}", parameters[0], parameters[1]);
    println!("Iterations: {}", fit.iterations);
    println!("Final objective value: {:.6e}", fit.objective_function);
    Ok(())
}
```

## Installation

Add `lmopt` to your `Cargo.toml`:

```toml
[dependencies]
lmopt = "0.3.0"
anyhow = "1"
```

Enable only the interoperability you need:

```toml
[dependencies]
lmopt = { version = "0.3.0", features = ["ndarray", "nalgebra"] }
```

The advanced matrix API also requires a direct `faer` dependency so its types
can be named in your code. Avoid enabling unrelated faer defaults:

```toml
[dependencies]
lmopt = "0.3.0"
faer = { version = "0.22.6", default-features = false, features = ["std"] }
```

## Requirements

- Rust 1.85 or newer
- `ndarray` and `nalgebra` support are optional features

## Documentation

For detailed documentation and more examples, see the [API Documentation](https://docs.rs/lmopt).

## Usage Guide

### Advanced problem API

Use `LevenbergMarquardt` and `LeastSquaresProblem` when you need an analytical
Jacobian, non-`f64` scalars, or detailed configuration:

```rust
impl LeastSquaresProblem<f64> for MyProblem {
    // Required: Calculate residuals
    fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
        // Your implementation here...
    }
    
    // Optional: Provide analytical Jacobian (recommended for performance)
    fn jacobian(&self, parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
        // Your implementation here...
        // Return None to let the default Auto strategy use central differences
    }
    
}
```

### Jacobian Calculation Methods

You can choose from several methods for calculating the Jacobian matrix:

```rust
// Automatically use an analytical Jacobian when provided, otherwise central differences
let optimizer = LevenbergMarquardt::new()
    .with_jacobian_method(JacobianMethod::Auto);

// Use the user-provided analytical Jacobian (most efficient)
let optimizer = LevenbergMarquardt::new()
    .with_jacobian_method(JacobianMethod::UserProvided);

// Use numerical differentiation with central differences (most accurate numerical method)
let optimizer = LevenbergMarquardt::new()
    .with_jacobian_method(JacobianMethod::NumericalCentral)
    .with_numerical_diff_step_size(1e-6);

// Use faster but less accurate numerical methods
let optimizer = LevenbergMarquardt::new()
    .with_jacobian_method(JacobianMethod::NumericalForward); // or NumericalBackward
```

### Configuring the Optimizer

The library provides a fluent API for configuring the optimizer:

```rust
let optimizer = LevenbergMarquardt::new()
    // Set maximum number of iterations
    .with_max_iterations(200)
    
    // Set convergence tolerances
    .with_ftol(1e-8) // Relative objective-reduction tolerance
    .with_xtol(1e-8) // Relative scaled-parameter-step tolerance
    .with_gtol(1e-8) // Scaled-gradient tolerance
    
    // Set initial damping parameter
    .with_tau(1e-3)
    
    // Choose Jacobian calculation method
    .with_jacobian_method(JacobianMethod::NumericalCentral)
    
    // Override the precision-aware default finite-difference step
    .with_numerical_diff_step_size(1e-6);
```

### Analyzing Results

The `MinimizationReport` struct provides detailed information about the optimization:

```rust
// `minimize` returns NoConvergence instead of an apparently successful Result.
let result = optimizer.minimize(&problem, &initial_guess)?;
println!("Solution parameters: {:?}", result.parameters());
println!("Final objective value: {}", result.objective_function);
println!("Accepted/rejected steps: {}/{}", result.accepted_steps, result.rejected_steps);
println!("Residual/Jacobian evaluations: {}/{}", result.residual_evaluations, result.jacobian_evaluations);

// Use `optimize` when a partial report at the iteration limit is useful.
let report = optimizer.optimize(&problem, &initial_guess)?;
println!("Converged: {} ({:?})", report.converged(), report.termination_reason);
```

## Examples

The repository includes several examples:

- **[linear_fitting.rs](examples/linear_fitting.rs)**: Basic fitting of a line to data points
- **[gaussian_fitting.rs](examples/gaussian_fitting.rs)**: Fitting a Gaussian curve to data
- **[jacobian_methods.rs](examples/jacobian_methods.rs)**: Comparing different Jacobian calculation methods

Run the examples with:

```
cargo run --example linear_fitting
cargo run --example gaussian_fitting
cargo run --example jacobian_methods
```

## Performance Tips

For optimal performance:

1. **Provide an analytical Jacobian** when possible
2. Use **central differences** when an analytical Jacobian is unavailable
3. **Scale your parameters** appropriately to improve convergence
4. Take advantage of **faer's optimizations** for matrix operations

Reproducible Criterion benchmarks cover Jacobian construction and end-to-end
optimization at multiple problem sizes:

```text
cargo bench --bench performance
```

Treat the results as the baseline for performance changes; the example timing
output is intended for diagnostics, not benchmarking. The first measured
baseline and reproduction command are recorded in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Upgrading

Version 0.3 adds the closure API and changes convergence and non-convergence
handling. See [`docs/MIGRATING-0.3.md`](docs/MIGRATING-0.3.md) and
[`CHANGELOG.md`](CHANGELOG.md).

## License

This project is licensed under the MIT License; see `LICENSE` for details.

## Acknowledgments

- [faer](https://github.com/sarah-ek/faer-rs) for the high-performance linear algebra operations
- [levenberg-marquardt](https://github.com/srayagarwal/levenberg-marquardt) crate for algorithm design inspiration
- [lmfit-py](https://lmfit.github.io/lmfit-py/) for advanced fitting concepts
