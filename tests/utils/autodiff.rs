#![cfg(feature = "autodiff")]

use faer::Mat;
use lmopt::utils::jacobian::{get_jacobian_calculator, JacobianCalculator};
use lmopt::{JacobianMethod, LeastSquaresProblem, Result};
use std::autodiff::{Active, Context};

// Define a simple problem for testing autodiff
struct QuadraticProblem;

impl LeastSquaresProblem<f64> for QuadraticProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Residuals: [x^2, y^2, x*y]
        // These are nonlinear functions with well-known derivatives
        let mut residuals = Mat::zeros(3, 1);
        residuals[(0, 0)] = x * x;
        residuals[(1, 0)] = y * y;
        residuals[(2, 0)] = x * y;

        Ok(residuals)
    }

    // Analytical Jacobian for comparison
    fn jacobian(&self, parameters: &Mat<f64>) -> Option<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Jacobian:
        // [2x, 0 ]
        // [0,  2y]
        // [y,  x ]
        let mut jacobian = Mat::zeros(3, 2);
        jacobian[(0, 0)] = 2.0 * x;
        jacobian[(0, 1)] = 0.0;
        jacobian[(1, 0)] = 0.0;
        jacobian[(1, 1)] = 2.0 * y;
        jacobian[(2, 0)] = y;
        jacobian[(2, 1)] = x;

        Some(jacobian)
    }
}

#[test]
fn test_autodiff_jacobian() {
    let problem = QuadraticProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i.0 == 0 { 2.0 } else { 3.0 });

    // Get the autodiff calculator
    let calculator = get_jacobian_calculator::<f64>(JacobianMethod::AutoDiff, 0.0);

    // Calculate the Jacobian using autodiff
    let autodiff_jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

    // Get the analytical Jacobian
    let analytical_jacobian = problem.jacobian(&params).unwrap();

    // Check dimensions
    assert_eq!(autodiff_jacobian.nrows(), 3);
    assert_eq!(autodiff_jacobian.ncols(), 2);

    // Compare the results with high precision (autodiff should be exact for polynomials)
    for i in 0..3 {
        for j in 0..2 {
            let diff = (autodiff_jacobian[(i, j)] - analytical_jacobian[(i, j)]).abs();
            assert!(
                diff < 1e-10,
                "Large difference at ({}, {}): autodiff = {}, analytical = {}",
                i,
                j,
                autodiff_jacobian[(i, j)],
                analytical_jacobian[(i, j)]
            );
        }
    }

    // Verify the method used
    assert_eq!(calculator.method_used(), JacobianMethod::AutoDiff);
}

// Define a more complex problem with trigonometric functions
struct TrigProblem;

impl LeastSquaresProblem<f64> for TrigProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Residuals with trigonometric functions
        let mut residuals = Mat::zeros(2, 1);
        residuals[(0, 0)] = x.sin() + y.cos();
        residuals[(1, 0)] = x.cos() * y.sin();

        Ok(residuals)
    }

    // Analytical Jacobian for comparison
    fn jacobian(&self, parameters: &Mat<f64>) -> Option<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Jacobian:
        // [cos(x), -sin(y)]
        // [-sin(x)*sin(y), cos(x)*cos(y)]
        let mut jacobian = Mat::zeros(2, 2);
        jacobian[(0, 0)] = x.cos();
        jacobian[(0, 1)] = -y.sin();
        jacobian[(1, 0)] = -x.sin() * y.sin();
        jacobian[(1, 1)] = x.cos() * y.cos();

        Some(jacobian)
    }
}

#[test]
fn test_autodiff_with_trig() {
    let problem = TrigProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i.0 == 0 { 0.5 } else { 1.2 });

    // Calculate the Jacobian using autodiff
    let calculator = get_jacobian_calculator::<f64>(JacobianMethod::AutoDiff, 0.0);
    let autodiff_jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

    // Get the analytical Jacobian
    let analytical_jacobian = problem.jacobian(&params).unwrap();

    // Check dimensions
    assert_eq!(autodiff_jacobian.nrows(), 2);
    assert_eq!(autodiff_jacobian.ncols(), 2);

    // Compare the results (allow slightly larger error due to transcendental functions)
    for i in 0..2 {
        for j in 0..2 {
            let diff = (autodiff_jacobian[(i, j)] - analytical_jacobian[(i, j)]).abs();
            assert!(
                diff < 1e-8,
                "Large difference at ({}, {}): autodiff = {}, analytical = {}",
                i,
                j,
                autodiff_jacobian[(i, j)],
                analytical_jacobian[(i, j)]
            );
        }
    }
}
