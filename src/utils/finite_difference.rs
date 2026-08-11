use crate::{Error, LeastSquaresProblem, Result};
use faer::Mat;
use faer_traits::RealField;
use num_traits::Float;

/// Calculates a Jacobian matrix using finite differences.
///
/// `step_size` is relative to each parameter's magnitude, with an absolute
/// floor of `step_size` for parameters whose magnitude is less than one.
pub fn calculate_jacobian<T, P>(problem: &P, parameters: &Mat<T>, step_size: T, method: FiniteDifferenceMethod) -> Result<Mat<T>>
where
    T: RealField + Copy + Float,
    P: LeastSquaresProblem<T>,
{
    let residuals = problem.residuals(parameters)?;
    calculate_jacobian_with_residuals(parameters, &residuals, step_size, method, |perturbed| problem.residuals(perturbed))
}

pub(crate) fn calculate_jacobian_with_residuals<T, F>(parameters: &Mat<T>, residuals: &Mat<T>, relative_step: T, method: FiniteDifferenceMethod, evaluate: F) -> Result<Mat<T>>
where
    T: RealField + Copy + Float,
    F: Fn(&Mat<T>) -> Result<Mat<T>>,
{
    if !Float::is_finite(relative_step) || relative_step <= T::zero() {
        return Err(Error::InvalidParameter("numerical differentiation step size must be finite and greater than zero".to_string()));
    }

    let n_params = parameters.nrows();
    let n_residuals = residuals.nrows();
    validate_residuals(residuals, n_residuals)?;

    let mut jacobian = Mat::zeros(n_residuals, n_params);
    let mut perturbed_params = parameters.clone();

    for j in 0..n_params {
        let original_value = parameters[(j, 0)];
        let step = relative_step * Float::max(Float::abs(original_value), T::one());

        match method {
            FiniteDifferenceMethod::Forward => {
                perturbed_params[(j, 0)] = original_value + step;
                let forward = evaluate(&perturbed_params)?;
                validate_residuals(&forward, n_residuals)?;
                for i in 0..n_residuals {
                    jacobian[(i, j)] = (forward[(i, 0)] - residuals[(i, 0)]) / step;
                }
            }
            FiniteDifferenceMethod::Central => {
                perturbed_params[(j, 0)] = original_value + step;
                let forward = evaluate(&perturbed_params)?;
                validate_residuals(&forward, n_residuals)?;

                perturbed_params[(j, 0)] = original_value - step;
                let backward = evaluate(&perturbed_params)?;
                validate_residuals(&backward, n_residuals)?;

                for i in 0..n_residuals {
                    jacobian[(i, j)] = (forward[(i, 0)] - backward[(i, 0)]) / (step + step);
                }
            }
            FiniteDifferenceMethod::Backward => {
                perturbed_params[(j, 0)] = original_value - step;
                let backward = evaluate(&perturbed_params)?;
                validate_residuals(&backward, n_residuals)?;
                for i in 0..n_residuals {
                    jacobian[(i, j)] = (residuals[(i, 0)] - backward[(i, 0)]) / step;
                }
            }
        }

        perturbed_params[(j, 0)] = original_value;
    }

    Ok(jacobian)
}

fn validate_residuals<T>(residuals: &Mat<T>, expected_rows: usize) -> Result<()>
where
    T: RealField + Copy + Float,
{
    if residuals.nrows() != expected_rows || residuals.ncols() != 1 {
        return Err(Error::DimensionMismatch(format!(
            "Residual dimensions changed during numerical differentiation: got {}x{}, expected {}x1",
            residuals.nrows(),
            residuals.ncols(),
            expected_rows
        )));
    }
    for i in 0..expected_rows {
        if !Float::is_finite(residuals[(i, 0)]) {
            return Err(Error::Numerical(format!("Residual element {i} is not finite during numerical differentiation")));
        }
    }
    Ok(())
}

/// Methods for finite difference approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDifferenceMethod {
    /// Forward difference: `(f(x+h) - f(x)) / h`.
    Forward,
    /// Central difference: `(f(x+h) - f(x-h)) / (2*h)`.
    Central,
    /// Backward difference: `(f(x) - f(x-h)) / h`.
    Backward,
}
