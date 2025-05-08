# lmopt - Levenberg-Marquardt Optimization with faer

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

A high-performance Rust implementation of the Levenberg-Marquardt algorithm for nonlinear least squares optimization, using the [faer](https://github.com/sarah-ek/faer-rs) linear algebra library.

## Features

- **Powerful Optimizer**: Robust implementation of the Levenberg-Marquardt algorithm with trust region strategy
- **High Performance**: Built on the highly optimized `faer` library for efficient matrix operations
- **Multiple Jacobian Methods**:
  - User-provided analytical Jacobian
  - Automatic differentiation (using Rust's experimental autodiff feature)
  - Numerical differentiation (central, forward, or backward differences)
- **Matrix Interoperability**: Seamless conversion between `faer`, `ndarray`, and `nalgebra` matrices
- **Comprehensive API**: Fluent interface for configurability and ease of use
- **Error Handling**: Extensive error types with context using `thiserror` and `anyhow`

## Quick Start

```rust
use lmopt::{LeastSquaresProblem, LevenbergMarquardt, JacobianMethod, Result};
use faer::Col;

// Define a least squares problem: fitting a line y = a*x + b
struct LinearModel {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl LeastSquaresProblem<f64> for LinearModel {
    fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
        let a = parameters[(0, 0)];
        let b = parameters[(1, 0)];
        
        let n = self.x_data.len();
        let mut residuals = faer::Mat::zeros(n, 1);
        
        for i in 0..n {
            let x = self.x_data[i];
            let y = self.y_data[i];
            let predicted = a * x + b;
            // residual = predicted - observed
            residuals[(i, 0)] = predicted - y;
        }
        
        Ok(residuals)
    }
    
    fn jacobian(&self, _parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
        let n = self.x_data.len();
        let mut jacobian = faer::Mat::zeros(n, 2);
        
        for i in 0..n {
            let x = self.x_data[i];
            // ∂r_i/∂a = x_i
            jacobian[(i, 0)] = x;
            // ∂r_i/∂b = 1
            jacobian[(i, 1)] = 1.0;
        }
        
        Some(jacobian)
    }
}

fn main() -> Result<()> {
    // Create sample data for y = 2x + 3 with some noise
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![3.1, 5.2, 6.9, 9.1, 11.0, 13.1];
    
    let problem = LinearModel { x_data, y_data };
    
    // Initial guess for parameters
    let initial_guess = Col::from_vec(vec![1.0, 1.0]);
    
    // Create optimizer with custom settings
    let optimizer = LevenbergMarquardt::new()
        .with_max_iterations(100)
        .with_epsilon_1(1e-8)
        .with_epsilon_2(1e-8)
        .with_jacobian_method(JacobianMethod::UserProvided);
    
    // Solve the optimization problem
    let result = optimizer.minimize(&problem, &initial_guess)?;
    
    // Extract the optimized parameters
    let a = result.solution_params[(0, 0)];
    let b = result.solution_params[(1, 0)];
    
    println!("Fitted line: y = {:.4}*x + {:.4}", a, b);
    println!("Iterations: {}", result.iterations);
    println!("Final objective value: {:.6e}", result.objective_function);
    
    Ok(())
}
```

## Installation

Add `lmopt` to your `Cargo.toml`:

```toml
[dependencies]
lmopt = "0.1.0"
faer = "0.22.6"  # Required dependency
```

If you want to use the autodiff feature (requires nightly Rust):

```toml
[dependencies]
lmopt = { version = "0.1.0", features = ["autodiff"] }
faer = { version = "0.22.6", features = ["nightly"] }
```

## Requirements

- Rust (nightly recommended for autodiff feature)
- Dependencies:
  - faer = "0.22.6"
  - faer-ext = "0.6.0"
  - ndarray = "0.16.1" (for interoperability)
  - nalgebra = "0.33.2" (for interoperability)
  - thiserror = "2.0.12"
  - anyhow = "1.0.98"

## Documentation

For detailed documentation and more examples, see the [API Documentation](https://docs.rs/lmopt).

## Usage Guide

### Defining a Problem

To use the Levenberg-Marquardt algorithm, you need to define your least squares problem by implementing the `LeastSquaresProblem` trait:

```rust
impl LeastSquaresProblem<f64> for MyProblem {
    // Required: Calculate residuals
    fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
        // Your implementation here...
    }
    
    // Optional: Provide analytical Jacobian (recommended for performance)
    fn jacobian(&self, parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
        // Your implementation here...
        // Return None to use automatic or numerical differentiation
    }
    
    // Optional: Hint whether to use autodiff when no Jacobian is provided
    fn prefer_autodiff(&self) -> bool {
        true // Default is true
    }
}
```

### Jacobian Calculation Methods

You can choose from several methods for calculating the Jacobian matrix:

```rust
// Use the user-provided analytical Jacobian (most efficient)
let optimizer = LevenbergMarquardt::new()
    .with_jacobian_method(JacobianMethod::UserProvided);

// Use automatic differentiation (requires nightly and autodiff feature)
let optimizer = LevenbergMarquardt::new()
    .with_jacobian_method(JacobianMethod::AutoDiff);

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
    .with_epsilon_1(1e-8) // For relative reduction in residuals
    .with_epsilon_2(1e-8) // For relative change in parameters
    
    // Set initial damping parameter
    .with_tau(1e-3)
    
    // Choose Jacobian calculation method
    .with_jacobian_method(JacobianMethod::NumericalCentral)
    
    // Set step size for numerical differentiation
    .with_numerical_diff_step_size(1e-6);
```

### Analyzing Results

The `MinimizationReport` struct provides detailed information about the optimization:

```rust
let result = optimizer.minimize(&problem, &initial_guess)?;

// Check if the optimization was successful
if result.success {
    // Access the solution
    println!("Solution parameters: {:?}", result.solution_params);
    println!("Final objective value: {}", result.objective_function);
    println!("Residuals: {:?}", result.residuals);
    
    // Performance information
    println!("Iterations: {}", result.iterations);
    println!("Execution time: {:?}", result.execution_time);
    println!("Jacobian method used: {:?}", result.jacobian_method_used);
} else {
    // Analyze why optimization failed
    println!("Optimization failed: {:?}", result.termination_reason);
}
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
2. Use **automatic differentiation** for complex functions when analytical Jacobian is difficult
3. Use **central differences** when numerical differentiation is required
4. **Scale your parameters** appropriately to improve convergence
5. Take advantage of **faer's optimizations** for matrix operations

## Automatic Differentiation

This library supports Rust's experimental `std::autodiff` module for automatic Jacobian calculation. This provides:

- **Exact Derivatives**: No approximation errors like with numerical methods
- **Ease of Use**: No need to manually derive and implement complex Jacobians
- **Performance**: Often faster than numerical differentiation
- **Flexibility**: Works with arbitrary differentiable functions

To use autodiff:

1. Enable the feature in your Cargo.toml:
   ```toml
   lmopt = { version = "0.1.0", features = ["autodiff"] }
   ```

2. Use nightly Rust:
   ```
   rustup override set nightly
   ```

3. Choose autodiff in your optimizer:
   ```rust
   let optimizer = LevenbergMarquardt::new()
       .with_jacobian_method(JacobianMethod::AutoDiff);
   ```

See the [autodiff_example.rs](examples/autodiff_example.rs) for a complete demonstration.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [faer](https://github.com/sarah-ek/faer-rs) for the high-performance linear algebra operations
- [levenberg-marquardt](https://github.com/srayagarwal/levenberg-marquardt) crate for algorithm design inspiration
- [lmfit-py](https://lmfit.github.io/lmfit-py/) for advanced fitting concepts