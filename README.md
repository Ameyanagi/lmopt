# lmopt - Levenberg-Marquardt Optimization with faer

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

A Rust implementation of the Levenberg-Marquardt algorithm for nonlinear least-squares optimization, using the high-performance [faer](https://github.com/sarah-ek/faer-rs) linear algebra library.

## Overview

This library provides:

1. A complete implementation of the Levenberg-Marquardt algorithm compatible with the [levenberg-marquardt](https://docs.rs/levenberg-marquardt/) crate
2. Efficient matrix operations powered by the faer library
3. A tiered approach to Jacobian calculation:
   - User-provided custom Jacobian (fastest)
   - Automatic differentiation (when available)
   - Numerical differentiation (multiple methods)
4. Additional features from the popular lmfit-py project (parameter system, uncertainty calculations, etc.)

## Features

- **Tiered Jacobian Calculation**: Choose the most appropriate method for computing derivatives
- **Trust Region Algorithm**: Stable optimization even for ill-conditioned problems
- **Matrix Interoperability**: Works with faer, ndarray, and nalgebra matrices
- **Configurable**: Extensive options for tuning the algorithm behavior
- **Robust Fitting**: Support for outlier-resistant optimization

## Quick Start

```rust
use lmopt::{LevenbergMarquardt, Problem, Result};
use ndarray::{array, Array1, Array2};

// Define a problem to solve (y = a*x + b)
struct LinearProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl Problem for LinearProblem {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        let a = params[0];
        let b = params[1];
        
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(x, y)| a * x + b - y)
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn parameter_count(&self) -> usize {
        2 // a and b
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
    
    // Optional: Provide analytical Jacobian for faster optimization
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        let n = self.x_data.len();
        let mut jac = Array2::zeros((n, 2));
        
        for i in 0..n {
            jac[[i, 0]] = self.x_data[i]; // ∂r/∂a = x
            jac[[i, 1]] = 1.0;            // ∂r/∂b = 1
        }
        
        Ok(jac)
    }
    
    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

fn main() -> Result<()> {
    // Create sample data for y = 2x + 3
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![5.1, 7.0, 9.1, 11.0, 13.1]; // With some noise
    
    let problem = LinearProblem { x_data: x, y_data: y };
    let initial_guess = array![1.0, 1.0];
    
    // Create optimizer and solve
    let optimizer = LevenbergMarquardt::new();
    let result = optimizer.minimize(&problem, initial_guess)?;
    
    println!("Solution: a = {:.4}, b = {:.4}", result.params[0], result.params[1]);
    println!("Cost: {:.6e}", result.cost);
    println!("Iterations: {}", result.iterations);
    println!("Jacobian method: {:?}", result.jacobian_method);
    
    Ok(())
}
```

## Requirements

- Rust (nightly required for autodiff features)
- Dependencies:
  - faer = "0.22" (high-performance linear algebra)
  - faer-ext = "0.6" (for matrix conversions)
  - ndarray = "0.15" (for compatibility with other libraries)
  - thiserror = "1.0" (for error handling)

## Documentation

For more detailed documentation and examples, see the [API Documentation](https://docs.rs/lmopt).

## Examples

The repository includes various examples demonstrating different features:

- [Basic Fitting](examples/model_fitting.rs)
- [Jacobian Methods](examples/jacobian_methods.rs)
- [Composite Models](examples/composite_models.rs)
- [Global Optimization](examples/global_optimization.rs)

## License

This project is licensed under the MIT License - see the LICENSE file for details.