use crate::lm::TerminationReason;
use faer_traits::RealField;

/// Check if the algorithm has converged based on various criteria
pub(crate) fn check_convergence<T>(old_residuals_norm: T, new_residuals_norm: T, step_norm: T, params_norm: T, epsilon_1: T, epsilon_2: T) -> Option<TerminationReason>
where
    T: RealField + Copy,
{
    // Check convergence based on relative reduction in residuals
    if old_residuals_norm > T::zero() {
        let relative_reduction = (old_residuals_norm - new_residuals_norm) / old_residuals_norm;
        if relative_reduction < epsilon_1 {
            return Some(TerminationReason::SmallRelativeReduction);
        }
    }

    // Check convergence based on relative change in parameters
    if params_norm > T::zero() {
        let relative_change = step_norm / params_norm;
        if relative_change < epsilon_2 {
            return Some(TerminationReason::SmallParameters);
        }
    }

    None
}
