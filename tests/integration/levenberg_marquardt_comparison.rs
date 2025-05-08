use faer::Mat;
use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};

// This test will compare our implementation with the original levenberg-marquardt crate
// It will be expanded once the core implementation is complete

// Rosenbrock function
struct RosenbrockProblem;

impl LeastSquaresProblem<f64> for RosenbrockProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Residuals: [(1-x), 10(y-x²)]
        let mut residuals = Mat::zeros(2, 1);
        residuals[(0, 0)] = 1.0 - x;
        residuals[(1, 0)] = 10.0 * (y - x * x);

        Ok(residuals)
    }

    fn jacobian(&self, parameters: &Mat<f64>) -> Option<Mat<f64>> {
        let x = parameters[(0, 0)];

        // Jacobian:
        // [-1, 0]
        // [-20x, 10]
        let mut jacobian = Mat::zeros(2, 2);
        jacobian[(0, 0)] = -1.0;
        jacobian[(0, 1)] = 0.0;
        jacobian[(1, 0)] = -20.0 * x;
        jacobian[(1, 1)] = 10.0;

        Some(jacobian)
    }
}

#[test]
#[ignore] // Ignore until the LM algorithm is implemented
fn test_rosenbrock_optimization() {
    let problem = RosenbrockProblem;
    let initial_guess = Mat::zeros(2, 1); // Start at (0, 0)

    let optimizer = LevenbergMarquardt::new().with_max_iterations(100).with_jacobian_method(JacobianMethod::UserProvided);

    let result = optimizer.minimize(&problem, &initial_guess).unwrap();

    // Check that we found the minimum at (1, 1)
    assert!(result.success);
    assert!((result.solution_params[(0, 0)] - 1.0).abs() < 1e-6);
    assert!((result.solution_params[(1, 0)] - 1.0).abs() < 1e-6);
    assert!(result.objective_function < 1e-12);
}
