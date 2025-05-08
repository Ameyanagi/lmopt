use faer::Mat;
use lmopt::{LeastSquaresProblem, Result};

// Test problem: Rosenbrock function
// f(x,y) = (1-x)² + 100(y-x²)²
// Minimum at (1, 1)
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

    fn prefer_autodiff(&self) -> bool {
        // Prefer autodiff for this problem
        true
    }
}

#[test]
fn test_rosenbrock_residuals() {
    let problem = RosenbrockProblem;

    // Case 1: At the minimum (1, 1)
    let params = Mat::from_fn(2, 1, |_, _| 1.0);
    let residuals = problem.residuals(&params).unwrap();

    // Residuals should be nearly zero at the minimum
    assert_eq!(residuals.nrows(), 2);
    assert_eq!(residuals.ncols(), 1);
    assert!(residuals[(0, 0)].abs() < 1e-10);
    assert!(residuals[(1, 0)].abs() < 1e-10);

    // Case 2: At point (0, 0)
    let params = Mat::zeros(2, 1);
    let residuals = problem.residuals(&params).unwrap();

    // Expected residuals: [1, 0]
    assert_eq!(residuals.nrows(), 2);
    assert_eq!(residuals.ncols(), 1);
    assert!((residuals[(0, 0)] - 1.0).abs() < 1e-10);
    assert!(residuals[(1, 0)].abs() < 1e-10);

    // Case 3: At point (2, 3)
    let mut params = Mat::zeros(2, 1);
    params[(0, 0)] = 2.0;
    params[(1, 0)] = 3.0;
    let residuals = problem.residuals(&params).unwrap();

    // Expected residuals: [-1, 10(3-4)] = [-1, -10]
    assert_eq!(residuals.nrows(), 2);
    assert_eq!(residuals.ncols(), 1);
    assert!((residuals[(0, 0)] + 1.0).abs() < 1e-10);
    assert!((residuals[(1, 0)] + 10.0).abs() < 1e-10);
}

#[test]
fn test_rosenbrock_jacobian() {
    let problem = RosenbrockProblem;

    // Case 1: At point (1, 1)
    let params = Mat::from_fn(2, 1, |_, _| 1.0);
    let jacobian = problem.jacobian(&params).unwrap();

    // Expected Jacobian at (1, 1):
    // [-1, 0]
    // [-20, 10]
    assert_eq!(jacobian.nrows(), 2);
    assert_eq!(jacobian.ncols(), 2);
    assert_eq!(jacobian[(0, 0)], -1.0);
    assert_eq!(jacobian[(0, 1)], 0.0);
    assert_eq!(jacobian[(1, 0)], -20.0);
    assert_eq!(jacobian[(1, 1)], 10.0);

    // Case 2: At point (0, 0)
    let params = Mat::zeros(2, 1);
    let jacobian = problem.jacobian(&params).unwrap();

    // Expected Jacobian at (0, 0):
    // [-1, 0]
    // [0, 10]
    assert_eq!(jacobian.nrows(), 2);
    assert_eq!(jacobian.ncols(), 2);
    assert_eq!(jacobian[(0, 0)], -1.0);
    assert_eq!(jacobian[(0, 1)], 0.0);
    assert_eq!(jacobian[(1, 0)], 0.0);
    assert_eq!(jacobian[(1, 1)], 10.0);

    // Case 3: At point (2, 3)
    let mut params = Mat::zeros(2, 1);
    params[(0, 0)] = 2.0;
    params[(1, 0)] = 3.0;
    let jacobian = problem.jacobian(&params).unwrap();

    // Expected Jacobian at (2, 3):
    // [-1, 0]
    // [-40, 10]
    assert_eq!(jacobian.nrows(), 2);
    assert_eq!(jacobian.ncols(), 2);
    assert_eq!(jacobian[(0, 0)], -1.0);
    assert_eq!(jacobian[(0, 1)], 0.0);
    assert_eq!(jacobian[(1, 0)], -40.0);
    assert_eq!(jacobian[(1, 1)], 10.0);
}
