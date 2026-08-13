#![forbid(unsafe_code)]
#![deny(clippy::expect_used, clippy::unwrap_used)]

//! # lmopt: A Levenberg-Marquardt optimization library using faer
//!
//! `lmopt` implements the Levenberg-Marquardt algorithm for dense nonlinear
//! least-squares optimization, leveraging [faer](https://github.com/sarah-ek/faer-rs)
//! for linear algebra.
//!
//! ## Features
//!
//! - **Powerful Optimizer**: Robust Levenberg-Marquardt implementation with trust region strategy
//! - **Measured Performance**: Criterion baselines for Jacobians and complete solves
//! - **Robust Linear Solves**: Augmented column-pivoted QR with truncated-SVD fallback
//! - **Multiple Jacobian Methods**:
//!   - Automatic selection between analytical and numerical differentiation
//!   - User-provided custom analytical Jacobian
//!   - Numerical differentiation (forward, backward, or central differences)
//! - **Matrix Interoperability**: Optional `ndarray` and `nalgebra` conversion features
//! - **Error Handling**: Structured, matchable errors using `thiserror`
//!
//! ## Basic Usage
//!
//! ```rust
//! let xs = [0.0, 1.0, 2.0, 3.0];
//! let ys = [1.0, 3.0, 5.0, 7.0];
//! let fit = lmopt::least_squares(&[0.0, 0.0], |parameters| {
//!     let [slope, intercept] = parameters else { return Vec::new() };
//!     xs.iter().zip(ys).map(|(x, y)| slope * x + intercept - y).collect()
//! })?;
//!
//! assert!((fit.parameters()[0] - 2.0).abs() < 1e-6);
//! assert!((fit.parameters()[1] - 1.0).abs() < 1e-6);
//! # Ok::<(), lmopt::Error>(())
//! ```
//!
//! ## Jacobian Calculation
//!
//! The library provides several methods for calculating the Jacobian matrix:
//!
//! 1. **Auto**: Prefer an analytical Jacobian and otherwise use central differences
//! 2. **User-Provided**: The fastest and most accurate method when available
//! 3. **Numerical Differentiation**:
//!    - Forward differences: `(f(x+h) - f(x)) / h`
//!    - Central differences: `(f(x+h) - f(x-h)) / (2*h)` (more accurate)
//!    - Backward differences: `(f(x) - f(x-h)) / h`
//! ## Performance Considerations
//!
//! For optimal performance:
//!
//! - Provide an analytical Jacobian when possible
//! - Use central differences when numerical differentiation is required
//! - Scale your parameters appropriately to improve convergence
//! - For very large problems, consider the trust region approach's memory usage
//! - Use the Criterion harness (`cargo bench --bench performance`) for comparisons
//!
//! ## Advanced Features
//!
//! The library provides access to detailed information about the optimization process:
//!
//! - Iteration count and convergence reason
//! - Execution time statistics
//! - Final residuals and objective function value
//! - The method used for Jacobian calculation

mod easy;
mod error;
mod lm;
mod problem;
pub mod utils;

// Re-export faer_traits::RealField
pub use faer_traits::RealField;

// Re-export core functionality
pub use easy::{least_squares, try_least_squares};
pub use error::{Error, Result};
pub use lm::{JacobianMethod, LevenbergMarquardt, MinimizationReport, TerminationReason};
pub use problem::LeastSquaresProblem;

// Re-export utils for convenience
pub use utils::jacobian::JacobianCalculator;
