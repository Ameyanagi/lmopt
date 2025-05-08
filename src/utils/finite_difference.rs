use crate::{LeastSquaresProblem, Result};
use faer::Mat;
use faer_traits::RealField;
use num_traits::FromPrimitive;

/// Calculates a Jacobian matrix using finite difference approximation.
pub fn calculate_jacobian<T, P>(problem: &P, parameters: &Mat<T>, step_size: T, method: FiniteDifferenceMethod) -> Result<Mat<T>>
where
    T: RealField + Copy + FromPrimitive,
    P: LeastSquaresProblem<T>,
{
    let residuals = problem.residuals(parameters)?;
    let n_params = parameters.nrows();
    let n_residuals = residuals.nrows();

    let mut jacobian = Mat::zeros(n_residuals, n_params);
    let mut perturbed_params = parameters.clone();

    for j in 0..n_params {
        match method {
            FiniteDifferenceMethod::Forward => {
                // Forward difference: (f(x+h) - f(x)) / h
                let original_val = parameters[(j, 0)];
                perturbed_params[(j, 0)] = original_val + step_size;

                let perturbed_residuals = problem.residuals(&perturbed_params)?;

                // Calculate derivative
                for i in 0..n_residuals {
                    jacobian[(i, j)] = (perturbed_residuals[(i, 0)] - residuals[(i, 0)]) / step_size;
                }

                // Restore original value
                perturbed_params[(j, 0)] = original_val;
            }
            FiniteDifferenceMethod::Central => {
                // Central difference: (f(x+h) - f(x-h)) / (2*h)
                let original_val = parameters[(j, 0)];

                // f(x+h)
                perturbed_params[(j, 0)] = original_val + step_size;
                let forward_residuals = problem.residuals(&perturbed_params)?;

                // f(x-h)
                perturbed_params[(j, 0)] = original_val - step_size;
                let backward_residuals = problem.residuals(&perturbed_params)?;

                // Calculate derivative
                for i in 0..n_residuals {
                    jacobian[(i, j)] = (forward_residuals[(i, 0)] - backward_residuals[(i, 0)]) / (step_size + step_size);
                }

                // Restore original value
                perturbed_params[(j, 0)] = original_val;
            }
            FiniteDifferenceMethod::Backward => {
                // Backward difference: (f(x) - f(x-h)) / h
                let original_val = parameters[(j, 0)];
                perturbed_params[(j, 0)] = original_val - step_size;

                let perturbed_residuals = problem.residuals(&perturbed_params)?;

                // Calculate derivative
                for i in 0..n_residuals {
                    jacobian[(i, j)] = (residuals[(i, 0)] - perturbed_residuals[(i, 0)]) / step_size;
                }

                // Restore original value
                perturbed_params[(j, 0)] = original_val;
            }
        }
    }

    Ok(jacobian)
}

/// Methods for finite difference approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDifferenceMethod {
    /// Forward difference: (f(x+h) - f(x)) / h
    Forward,
    /// Central difference: (f(x+h) - f(x-h)) / (2*h)
    Central,
    /// Backward difference: (f(x) - f(x-h)) / h
    Backward,
}
