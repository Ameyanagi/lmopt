use crate::lm::TerminationReason;
use faer_traits::RealField;

/// Check if the algorithm has converged based on various criteria
pub(crate) fn check_convergence<T>(old_objective: T, new_objective: T, scaled_step_norm: T, scaled_params_norm: T, ftol: T, xtol: T) -> Option<TerminationReason>
where
    T: RealField + Copy,
{
    // A worsening trial point must never be interpreted as convergence.
    if old_objective > T::zero() && new_objective <= old_objective {
        let relative_reduction = (old_objective - new_objective) / old_objective;
        if relative_reduction <= ftol {
            return Some(TerminationReason::SmallRelativeReduction);
        }
    }

    // Keep this purely relative. An absolute floor in scaled coordinates makes
    // convergence depend on the arbitrary scale of the residuals/Jacobian.
    // Stationary solutions at the origin are handled by the gradient test.
    if scaled_params_norm > T::zero() && scaled_step_norm <= xtol * scaled_params_norm {
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
    fn a_small_step_at_the_origin_needs_the_gradient_test() {
        assert!(check_convergence(1.0, 0.5, 1e-20, 0.0, 1e-8, 1e-8).is_none());
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
