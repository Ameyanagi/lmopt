use crate::lm::TerminationReason;
use faer_traits::RealField;

/// Check if the algorithm has converged based on various criteria
pub(crate) fn check_convergence<T>(old_residuals_norm: T, new_residuals_norm: T, step_norm: T, params_norm: T, epsilon_1: T, epsilon_2: T) -> Option<TerminationReason>
where
    T: RealField + Copy,
{
    // A worsening trial point must never be interpreted as convergence.
    if old_residuals_norm > T::zero() && new_residuals_norm <= old_residuals_norm {
        let relative_reduction = (old_residuals_norm - new_residuals_norm) / old_residuals_norm;
        if relative_reduction <= epsilon_1 {
            return Some(TerminationReason::SmallRelativeReduction);
        }
    }

    // Include an absolute component so a solution near the origin can converge.
    if step_norm <= epsilon_2 * (params_norm + epsilon_2) {
        return Some(TerminationReason::SmallParameters);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::check_convergence;

    #[test]
    fn an_increase_is_not_convergence() {
        assert!(check_convergence(1.0, 1.1, 0.1, 1.0, 1e-8, 1e-8).is_none());
    }

    #[test]
    fn a_small_step_at_the_origin_can_converge() {
        assert!(check_convergence(1.0, 0.5, 1e-20, 0.0, 1e-8, 1e-8).is_some());
    }

    #[test]
    fn a_small_positive_reduction_can_converge() {
        assert!(check_convergence(1.0, 0.9999, 1.0, 1.0, 1e-3, 1e-8).is_some());
    }

    #[test]
    fn a_meaningful_reduction_and_step_continue() {
        assert!(check_convergence(1.0, 0.8, 0.1, 1.0, 1e-8, 1e-8).is_none());
    }
}
