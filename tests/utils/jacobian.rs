use faer::Mat;
use lmopt::utils::jacobian::{get_jacobian_calculator, JacobianCalculator};
use lmopt::{JacobianMethod, LeastSquaresProblem, Result};

// Define a simple problem for testing
struct LinearProblem;

impl LeastSquaresProblem<f64> for LinearProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let a = parameters[(0, 0)];
        let b = parameters[(1, 0)];

        // Linear function: f(x) = a*x + b
        // Residuals for x = [1, 2, 3]: [a+b-2, 2a+b-3, 3a+b-4]
        let mut residuals = Mat::zeros(3, 1);
        residuals[(0, 0)] = a + b - 2.0;
        residuals[(1, 0)] = 2.0 * a + b - 3.0;
        residuals[(2, 0)] = 3.0 * a + b - 4.0;

        Ok(residuals)
    }

    fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
        // Jacobian is constant for linear function:
        // [1, 1]
        // [2, 1]
        // [3, 1]
        let mut jacobian = Mat::zeros(3, 2);
        jacobian[(0, 0)] = 1.0;
        jacobian[(0, 1)] = 1.0;
        jacobian[(1, 0)] = 2.0;
        jacobian[(1, 1)] = 1.0;
        jacobian[(2, 0)] = 3.0;
        jacobian[(2, 1)] = 1.0;

        Some(jacobian)
    }
}

#[test]
fn test_user_provided_jacobian() {
    let problem = LinearProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i.0 == 0 { 1.0 } else { 1.0 });

    let calculator = get_jacobian_calculator::<f64>(JacobianMethod::UserProvided, 0.0);
    let jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

    // Verify the Jacobian
    assert_eq!(jacobian.nrows(), 3);
    assert_eq!(jacobian.ncols(), 2);

    // Check specific values
    assert_eq!(jacobian[(0, 0)], 1.0);
    assert_eq!(jacobian[(0, 1)], 1.0);
    assert_eq!(jacobian[(1, 0)], 2.0);
    assert_eq!(jacobian[(1, 1)], 1.0);
    assert_eq!(jacobian[(2, 0)], 3.0);
    assert_eq!(jacobian[(2, 1)], 1.0);

    // Verify method used
    assert_eq!(calculator.method_used(), JacobianMethod::UserProvided);
}

#[test]
fn test_numerical_central_jacobian() {
    let problem = LinearProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i.0 == 0 { 1.0 } else { 1.0 });

    // Using central difference method
    let calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalCentral, 1e-6);
    let jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

    // Verify the Jacobian
    assert_eq!(jacobian.nrows(), 3);
    assert_eq!(jacobian.ncols(), 2);

    // For a linear function, the numerical Jacobian should be very close to the analytical one
    assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-8);
    assert!((jacobian[(0, 1)] - 1.0).abs() < 1e-8);
    assert!((jacobian[(1, 0)] - 2.0).abs() < 1e-8);
    assert!((jacobian[(1, 1)] - 1.0).abs() < 1e-8);
    assert!((jacobian[(2, 0)] - 3.0).abs() < 1e-8);
    assert!((jacobian[(2, 1)] - 1.0).abs() < 1e-8);

    // Verify method used
    assert_eq!(calculator.method_used(), JacobianMethod::NumericalCentral);
}

#[test]
fn test_numerical_forward_jacobian() {
    let problem = LinearProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i.0 == 0 { 1.0 } else { 1.0 });

    // Using forward difference method
    let calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalForward, 1e-6);
    let jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

    // Verify the Jacobian
    assert_eq!(jacobian.nrows(), 3);
    assert_eq!(jacobian.ncols(), 2);

    // For a linear function, the numerical Jacobian should be very close to the analytical one
    assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-8);
    assert!((jacobian[(0, 1)] - 1.0).abs() < 1e-8);
    assert!((jacobian[(1, 0)] - 2.0).abs() < 1e-8);
    assert!((jacobian[(1, 1)] - 1.0).abs() < 1e-8);
    assert!((jacobian[(2, 0)] - 3.0).abs() < 1e-8);
    assert!((jacobian[(2, 1)] - 1.0).abs() < 1e-8);

    // Verify method used
    assert_eq!(calculator.method_used(), JacobianMethod::NumericalForward);
}

#[test]
fn test_fallback_with_autodiff() {
    let problem = LinearProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i.0 == 0 { 1.0 } else { 1.0 });

    // When AutoDiff is not available or fails, it should fall back to numerical differentiation
    let calculator = get_jacobian_calculator::<f64>(JacobianMethod::AutoDiff, 1e-6);

    #[cfg(feature = "autodiff")]
    {
        // This should emit an error that we're not testing
        let _result = calculator.calculate_jacobian(&problem, &params);
        assert_eq!(calculator.method_used(), JacobianMethod::AutoDiff);
    }

    #[cfg(not(feature = "autodiff"))]
    {
        let jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

        // Verify the Jacobian values (calculated using central difference)
        assert!((jacobian[(0, 0)] - 1.0).abs() < 1e-8);
        assert!((jacobian[(0, 1)] - 1.0).abs() < 1e-8);
        assert!((jacobian[(1, 0)] - 2.0).abs() < 1e-8);
        assert!((jacobian[(1, 1)] - 1.0).abs() < 1e-8);
        assert!((jacobian[(2, 0)] - 3.0).abs() < 1e-8);
        assert!((jacobian[(2, 1)] - 1.0).abs() < 1e-8);

        // When autodiff is not available, it should default to central difference
        assert_eq!(calculator.method_used(), JacobianMethod::NumericalCentral);
    }
}
