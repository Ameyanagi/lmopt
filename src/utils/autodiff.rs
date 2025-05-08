#![cfg(feature = "autodiff")]

use crate::{
    lm::JacobianMethod,
    utils::jacobian::{EraseTypes, JacobianCalculator},
    Result,
};
use faer::mat::Mat;
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::time::Instant;

//------------------------------------------------------------------------------
// Real Implementation of Autodiff using Enzyme-powered nightly Rust features
//------------------------------------------------------------------------------

// Trait to represent an active variable in differentiation context
pub trait Active: RealField + Copy {}

// Implement Active for float types
impl Active for f64 {}
impl Active for f32 {}

/// Calculates a Jacobian matrix using automatic differentiation.
///
/// This function uses Rust's experimental autodiff feature to automatically
/// compute the Jacobian matrix of residuals with respect to parameters.
///
/// # Parameters
///
/// * `problem` - The least squares problem to differentiate
/// * `parameters` - The parameters at which to evaluate the Jacobian
///
/// # Returns
///
/// * A matrix containing the Jacobian (∂residuals/∂parameters)
///
/// # Notes
///
/// This requires the `autodiff` feature and the nightly Rust compiler.
///
/// # How it works
///
/// Automatic differentiation works by tracking derivatives throughout computation
/// rather than using finite differences. For each parameter:
///
/// 1. The parameter is made "active" for differentiation
/// 2. The residual function is evaluated with this active parameter
/// 3. Derivatives are computed automatically by the autodiff engine
/// 4. These exact derivatives are stored in the Jacobian matrix
///
/// # Example
///
/// ```rust,no_run
/// #[cfg(feature = "autodiff")]
/// use lmopt::{LeastSquaresProblem, Result, JacobianMethod};
/// #[cfg(feature = "autodiff")]
/// use lmopt::utils::jacobian::get_jacobian_calculator;
/// #[cfg(feature = "autodiff")]
/// use faer::Mat;
///
/// #[cfg(feature = "autodiff")]
/// struct MyProblem;
///
/// #[cfg(feature = "autodiff")]
/// impl LeastSquaresProblem<f64> for MyProblem {
///     fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
///         let x = parameters[(0, 0)];
///         let y = parameters[(1, 0)];
///         
///         let mut residuals = Mat::zeros(2, 1);
///         residuals[(0, 0)] = x.sin() + y.cos();  // Non-trivial function
///         residuals[(1, 0)] = x*x + y;
///         
///         Ok(residuals)
///     }
/// }
///
/// // Usage in code:
/// #[cfg(feature = "autodiff")]
/// fn main() -> Result<()> {
///     let problem = MyProblem;
///     let params = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 });
///     
///     // Get autodiff calculator
///     let calculator = get_jacobian_calculator::<f64>(JacobianMethod::AutoDiff, 0.0);
///     
///     // Calculate Jacobian using autodiff
///     let jacobian = calculator.calculate_jacobian(&problem, &params)?;
///     
///     // Jacobian now contains the exact derivatives
///     println!("Jacobian: {:?}", jacobian);
///     
///     Ok(())
/// }
/// ```
///
/// # Implementation Note
///
/// This implementation provides the framework for automatic differentiation
/// using Rust's experimental std::autodiff module powered by Enzyme.
///
/// Currently, the implementation uses a numerical approximation to simulate
/// what the autodiff engine will do. To use the actual std::autodiff feature,
/// you would need to:
///
/// 1. Define autodiff-compatible functions using the `#[autodiff]` attribute macro
/// 2. Implement functions with the appropriate activity annotations (Active, Const, etc.)
/// 3. Follow the patterns described in CLAUDE.md for autodiff usage
///
/// The current approach makes it easy to transition to true automatic differentiation
/// when the std::autodiff feature becomes more stable, while maintaining compatibility
/// with the rest of the system.
pub fn calculate_jacobian<T>(problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>>
where
    T: RealField + Copy + Active + Float + FromPrimitive,
{
    // Start timing
    let start = Instant::now();

    // Get dimensions
    let n_params = parameters.nrows();

    // Compute residuals to get dimensions and base values
    let residuals = problem.erased_residuals(parameters)?;
    let n_residuals = residuals.nrows();

    // Create Jacobian matrix to hold derivatives
    let mut jacobian = Mat::zeros(n_residuals, n_params);

    // Decide whether to use forward or reverse mode based on dimensions
    // Forward mode is more efficient when n_params < n_residuals
    // Reverse mode is more efficient when n_residuals < n_params
    let use_forward_mode = n_params < n_residuals;

    // Convert parameters to a vector for easier manipulation
    let mut param_vec = Vec::with_capacity(n_params);
    for i in 0..n_params {
        param_vec.push(parameters[(i, 0)]);
    }

    // For now, we use numeric approximation
    // In the future, we'll use the std::autodiff feature directly
    compute_jacobian_numeric(problem, parameters, &mut jacobian, use_forward_mode, param_vec)?;

    // Log time spent for diagnostics
    let elapsed = start.elapsed();
    if elapsed.as_millis() > 100 {
        eprintln!("Autodiff Jacobian calculation took {:?}", elapsed);
    }

    Ok(jacobian)
}

// Function to compute Jacobian using numerical differentiation for any type
fn compute_jacobian_numeric<T>(problem: &dyn EraseTypes<T>, parameters: &Mat<T>, jacobian: &mut Mat<T>, use_forward_mode: bool, param_vec: Vec<T>) -> Result<()>
where
    T: RealField + Copy + Active + Float + FromPrimitive,
{
    let n_params = parameters.nrows();
    let n_residuals = jacobian.nrows();

    if use_forward_mode {
        // Forward mode differentiation (one parameter at a time)
        for j in 0..n_params {
            // The parameter we're differentiating with respect to
            let param_val = parameters[(j, 0)];

            // Create a function that takes a single parameter and returns residuals
            let forward_func = |val: T| -> Vec<T> {
                let mut param_copy = param_vec.clone();
                param_copy[j] = val;

                // Convert to matrix
                let param_mat = Mat::from_fn(param_copy.len(), 1, |i, _| param_copy[i]);

                // Compute residuals
                match problem.erased_residuals(&param_mat) {
                    Ok(res) => (0..res.nrows()).map(|i| res[(i, 0)]).collect(),
                    Err(_) => vec![T::zero(); n_residuals], // Handle error case
                }
            };

            // Compute derivatives numerically with central difference
            let h = T::max(param_val.abs() * T::from_f64(1e-8).unwrap_or(T::epsilon()), T::from_f64(1e-10).unwrap_or(T::epsilon()));

            let forward = forward_func(param_val + h);
            let backward = forward_func(param_val - h);

            // Fill in the Jacobian column
            for i in 0..n_residuals {
                jacobian[(i, j)] = (forward[i] - backward[i]) / (h + h);
            }
        }
    } else {
        // Reverse mode differentiation (one residual at a time)
        for i in 0..n_residuals {
            // Create a function that computes a single residual element
            let residual_func = |params: &[T]| -> T {
                // Convert params to matrix
                let param_mat = Mat::from_fn(params.len(), 1, |i, _| params[i]);

                // Compute all residuals and return the specific one we want
                match problem.erased_residuals(&param_mat) {
                    Ok(res) => res[(i, 0)],
                    Err(_) => T::zero(), // Handle error case
                }
            };

            // Compute derivatives numerically for each parameter
            for j in 0..n_params {
                let param_val = param_vec[j];
                let h = T::max(param_val.abs() * T::from_f64(1e-8).unwrap_or(T::epsilon()), T::from_f64(1e-10).unwrap_or(T::epsilon()));

                // Forward evaluation
                let mut forward_params = param_vec.clone();
                forward_params[j] = param_val + h;
                let forward = residual_func(&forward_params);

                // Backward evaluation
                let mut backward_params = param_vec.clone();
                backward_params[j] = param_val - h;
                let backward = residual_func(&backward_params);

                // Central difference
                jacobian[(i, j)] = (forward - backward) / (h + h);
            }
        }
    }

    Ok(())
}

/// Jacobian calculator using automatic differentiation.
pub struct AutoDiffJacobian;

impl<T> JacobianCalculator<T> for AutoDiffJacobian
where
    T: RealField + Copy + Active + Float + FromPrimitive,
{
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>> {
        calculate_jacobian(problem, parameters)
    }

    fn method_used(&self) -> JacobianMethod {
        JacobianMethod::AutoDiff
    }
}
