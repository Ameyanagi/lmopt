use faer::Mat;
use lmopt::lm::trust_region::{adjust_lambda, calculate_parameter_update};

#[test]
fn test_parameter_update_calculation() {
    // Create a simple test problem: linear residuals
    // r = [x - 1, y - 2]
    // J = [1 0]
    //     [0 1]

    // Parameters: [0, 0]
    // Residuals: [-1, -2]
    let residuals = Mat::from_vec(vec![-1.0, -2.0], 2, 1);

    // Jacobian for this linear problem
    let mut jacobian = Mat::zeros(2, 2);
    jacobian[(0, 0)] = 1.0;
    jacobian[(1, 1)] = 1.0;

    // Unit diagonal (no scaling)
    let diag = Mat::ones(2, 1);

    // Test with lambda = 0
    let lambda = 0.0;
    let update = calculate_parameter_update(&jacobian, &residuals, lambda, &diag).unwrap();

    // With lambda = 0, we expect step = [1, 2]
    assert_eq!(update.step.nrows(), 2);
    assert!((update.step[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((update.step[(1, 0)] - 2.0).abs() < 1e-10);

    // Predicted reduction should be positive
    assert!(update.predicted_reduction > 0.0);

    // Test with lambda = 1
    let lambda = 1.0;
    let update = calculate_parameter_update(&jacobian, &residuals, lambda, &diag).unwrap();

    // With lambda = 1, we expect step to be smaller than [1, 2]
    assert_eq!(update.step.nrows(), 2);
    assert!(update.step[(0, 0)] > 0.0 && update.step[(0, 0)] < 1.0);
    assert!(update.step[(1, 0)] > 0.0 && update.step[(1, 0)] < 2.0);

    // Predicted reduction should be positive but smaller than before
    assert!(update.predicted_reduction > 0.0);
}

#[test]
fn test_lambda_adjustment() {
    // Test increasing lambda for failed steps
    let lambda = 1.0;
    let ratio = 0.0; // Doesn't matter for failed steps
    let success = false;

    let new_lambda = adjust_lambda(lambda, ratio, success);
    assert!(new_lambda > lambda); // Lambda should increase

    // Test increasing lambda for small ratio
    let lambda = 1.0;
    let ratio = 0.1; // Small ratio < 0.25
    let success = true;

    let new_lambda = adjust_lambda(lambda, ratio, success);
    assert!(new_lambda > lambda); // Lambda should increase

    // Test decreasing lambda for large ratio
    let lambda = 1.0;
    let ratio = 0.9; // Large ratio > 0.75
    let success = true;

    let new_lambda = adjust_lambda(lambda, ratio, success);
    assert!(new_lambda < lambda); // Lambda should decrease

    // Test keeping lambda the same for medium ratio
    let lambda = 1.0;
    let ratio = 0.5; // Medium ratio between 0.25 and 0.75
    let success = true;

    let new_lambda = adjust_lambda(lambda, ratio, success);
    assert_eq!(new_lambda, lambda); // Lambda should stay the same
}
