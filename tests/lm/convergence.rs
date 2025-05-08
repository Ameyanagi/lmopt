use lmopt::lm::convergence::check_convergence;
use lmopt::lm::TerminationReason;

#[test]
fn test_small_relative_reduction_convergence() {
    // Test case: Small relative reduction
    let old_residuals_norm = 1.0;
    let new_residuals_norm = 0.9999; // Very small reduction
    let step_norm = 1.0;
    let params_norm = 10.0;
    let epsilon_1 = 1e-3; // Relatively large tolerance for reduction
    let epsilon_2 = 1e-10; // Small tolerance for parameter change

    let termination = check_convergence(old_residuals_norm, new_residuals_norm, step_norm, params_norm, epsilon_1, epsilon_2);

    // Should converge due to small relative reduction
    assert!(termination.is_some());
    assert!(matches!(termination.unwrap(), TerminationReason::SmallRelativeReduction));
}

#[test]
fn test_small_parameters_convergence() {
    // Test case: Small relative parameter change
    let old_residuals_norm = 1.0;
    let new_residuals_norm = 0.9; // Significant reduction
    let step_norm = 1e-11; // Very small step
    let params_norm = 1.0;
    let epsilon_1 = 1e-10; // Small tolerance for reduction
    let epsilon_2 = 1e-10; // Small tolerance for parameter change

    let termination = check_convergence(old_residuals_norm, new_residuals_norm, step_norm, params_norm, epsilon_1, epsilon_2);

    // Should converge due to small parameter change
    assert!(termination.is_some());
    assert!(matches!(termination.unwrap(), TerminationReason::SmallParameters));
}

#[test]
fn test_no_convergence() {
    // Test case: No convergence criteria met
    let old_residuals_norm = 1.0;
    let new_residuals_norm = 0.8; // Significant reduction
    let step_norm = 0.1; // Significant step
    let params_norm = 1.0;
    let epsilon_1 = 1e-10; // Small tolerance for reduction
    let epsilon_2 = 1e-10; // Small tolerance for parameter change

    let termination = check_convergence(old_residuals_norm, new_residuals_norm, step_norm, params_norm, epsilon_1, epsilon_2);

    // Should not converge
    assert!(termination.is_none());
}

#[test]
fn test_zero_params_norm() {
    // Test case: Params norm is zero (can happen with all zeros)
    let old_residuals_norm = 1.0;
    let new_residuals_norm = 0.8; // Significant reduction
    let step_norm = 0.1; // Significant step
    let params_norm = 0.0; // Zero norm
    let epsilon_1 = 1e-10; // Small tolerance for reduction
    let epsilon_2 = 1e-10; // Small tolerance for parameter change

    let termination = check_convergence(old_residuals_norm, new_residuals_norm, step_norm, params_norm, epsilon_1, epsilon_2);

    // Should not converge due to smallParameters, since we can't compute the relative change
    assert!(termination.is_none());
}
