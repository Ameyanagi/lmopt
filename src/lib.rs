//! # lmopt: A Levenberg-Marquardt optimization library using faer
//!
//! `lmopt` is a high-performance implementation of the Levenberg-Marquardt algorithm
//! for nonlinear least squares optimization, leveraging the [faer](https://github.com/sarah-ek/faer-rs)
//! linear algebra library for efficient matrix operations.
//!
//! ## Features
//!
//! - **Powerful Optimizer**: Robust Levenberg-Marquardt implementation with trust region strategy
//! - **High Performance**: Built on the `faer` library for fast matrix operations
//! - **Multiple Jacobian Methods**:
//!   - User-provided custom analytical Jacobian
//!   - Automatic differentiation (with the `autodiff` feature and nightly Rust)
//!   - Numerical differentiation (forward, backward, or central differences)
//! - **Matrix Interoperability**: Seamless conversion between `faer`, `ndarray`, and `nalgebra` matrices
//! - **Error Handling**: Comprehensive error types with context using `thiserror` and `anyhow`
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use lmopt::{LeastSquaresProblem, LevenbergMarquardt, JacobianMethod, Result};
//!
//! // Define your problem by implementing the LeastSquaresProblem trait
//! struct MyProblem {
//!     // Your problem data here
//! }
//!
//! impl LeastSquaresProblem<f64> for MyProblem {
//!     // Compute the vector of residuals
//!     fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
//!         // Your residual calculation here
//!         todo!()
//!     }
//!
//!     // Optionally provide an analytical Jacobian
//!     fn jacobian(&self, parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
//!         // Your Jacobian calculation here, or return None to use automatic/numerical differentiation
//!         None
//!     }
//! }
//!
//! // Solve the optimization problem
//! fn main() -> Result<()> {
//!     let problem = MyProblem { /* ... */ };
//!     let initial_guess = faer::Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 }); // Starting parameters
//!
//!     // Configure the optimizer
//!     let optimizer = LevenbergMarquardt::new()
//!         .with_max_iterations(100)
//!         .with_epsilon_1(1e-8)
//!         .with_epsilon_2(1e-8)
//!         .with_jacobian_method(JacobianMethod::NumericalCentral);
//!
//!     // Run the optimization
//!     let result = optimizer.minimize(&problem, &initial_guess)?;
//!
//!     println!("Optimization succeeded: {}", result.success);
//!     println!("Solution parameters: {:?}", result.solution_params);
//!     println!("Final objective value: {}", result.objective_function);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Jacobian Calculation
//!
//! The library provides several methods for calculating the Jacobian matrix:
//!
//! 1. **User-Provided**: The fastest and most accurate method when available
//! 2. **Automatic Differentiation**: Using Rust's experimental `std::autodiff` feature (nightly only)
//!    - Exact derivatives without numerical approximation errors
//!    - Requires the nightly compiler and the `autodiff` feature flag
//!    - Ideal for complex functions where deriving manual Jacobians is difficult
//!    - Generally more accurate and often faster than numerical methods
//! 3. **Numerical Differentiation**:
//!    - Forward differences: `f(x+h) - f(x) / h`
//!    - Central differences: `f(x+h) - f(x-h) / (2*h)` (more accurate)
//!    - Backward differences: `f(x) - f(x-h) / h`
//!
//! ## Performance Considerations
//!
//! For optimal performance:
//!
//! - Provide an analytical Jacobian when possible
//! - Use central differences when numerical differentiation is required
//! - Scale your parameters appropriately to improve convergence
//! - For very large problems, consider the trust region approach's memory usage
//!
//! ## Advanced Features
//!
//! The library provides access to detailed information about the optimization process:
//!
//! - Iteration count and convergence reason
//! - Execution time statistics
//! - Final residuals and objective function value
//! - The method used for Jacobian calculation

// Enable experimental autodiff feature only when the autodiff feature is enabled
#![cfg_attr(feature = "autodiff", feature(autodiff))]

mod error;
mod lm;
mod problem;
pub mod utils;

// Re-export faer_traits::RealField
pub use faer_traits::RealField;

// Re-export core functionality
pub use error::{Error, Result};
pub use lm::{JacobianMethod, LevenbergMarquardt, MinimizationReport, TerminationReason};
pub use problem::LeastSquaresProblem;

// Re-export utils for convenience
pub use utils::jacobian::JacobianCalculator;
