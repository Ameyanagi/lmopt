use anyhow::{anyhow, Result as TestResult};
use faer::Mat;
use lmopt::utils::finite_difference::{calculate_jacobian, FiniteDifferenceMethod};
use lmopt::{LeastSquaresProblem, Result};

// Define a simple problem for testing
struct QuadraticProblem;

impl LeastSquaresProblem<f64> for QuadraticProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Residuals: [x^2, y^2, x*y]
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
fn test_forward_difference() -> TestResult<()> {
    let problem = QuadraticProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { 3.0 });

    let step_size = 1e-6;
    let numerical_jacobian = calculate_jacobian(&problem, &params, step_size, FiniteDifferenceMethod::Forward)?;
    let analytical_jacobian = problem.jacobian(&params).ok_or_else(|| anyhow!("analytical Jacobian is missing"))?;

    // Check that numerical approximation is close to analytical
    for i in 0..3 {
        for j in 0..2 {
            let rel_error = (numerical_jacobian[(i, j)] - analytical_jacobian[(i, j)]).abs() / analytical_jacobian[(i, j)].abs().max(1e-12);
            assert!(rel_error < 1e-5, "Forward difference rel error too large at ({}, {}): {}", i, j, rel_error);
        }
    }
    Ok(())
}

#[test]
fn test_central_difference() -> TestResult<()> {
    let problem = QuadraticProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { 3.0 });

    let step_size = 1e-6;
    let numerical_jacobian = calculate_jacobian(&problem, &params, step_size, FiniteDifferenceMethod::Central)?;
    let analytical_jacobian = problem.jacobian(&params).ok_or_else(|| anyhow!("analytical Jacobian is missing"))?;

    // Central difference should be more accurate than forward difference
    for i in 0..3 {
        for j in 0..2 {
            let rel_error = (numerical_jacobian[(i, j)] - analytical_jacobian[(i, j)]).abs() / analytical_jacobian[(i, j)].abs().max(1e-12);
            assert!(rel_error < 1e-8, "Central difference rel error too large at ({}, {}): {}", i, j, rel_error);
        }
    }
    Ok(())
}

#[test]
fn test_backward_difference() -> TestResult<()> {
    let problem = QuadraticProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { 3.0 });

    let step_size = 1e-6;
    let numerical_jacobian = calculate_jacobian(&problem, &params, step_size, FiniteDifferenceMethod::Backward)?;
    let analytical_jacobian = problem.jacobian(&params).ok_or_else(|| anyhow!("analytical Jacobian is missing"))?;

    // Check that numerical approximation is close to analytical
    for i in 0..3 {
        for j in 0..2 {
            let rel_error = (numerical_jacobian[(i, j)] - analytical_jacobian[(i, j)]).abs() / analytical_jacobian[(i, j)].abs().max(1e-12);
            assert!(rel_error < 1e-5, "Backward difference rel error too large at ({}, {}): {}", i, j, rel_error);
        }
    }
    Ok(())
}

struct ChangingResidualShape;

impl LeastSquaresProblem<f64> for ChangingResidualShape {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let rows = if parameters[(0, 0)] > 0.0 { 2 } else { 1 };
        Ok(Mat::zeros(rows, 1))
    }
}

#[test]
fn changing_residual_shape_returns_an_error() -> TestResult<()> {
    let error = match calculate_jacobian(&ChangingResidualShape, &Mat::zeros(1, 1), 1e-6, FiniteDifferenceMethod::Forward) {
        Ok(_) => return Err(anyhow!("changing residual shape unexpectedly succeeded")),
        Err(error) => error,
    };

    assert!(error.to_string().contains("dimensions changed"));
    Ok(())
}

#[test]
fn invalid_step_size_returns_an_error() -> TestResult<()> {
    let error = match calculate_jacobian(&QuadraticProblem, &Mat::ones(2, 1), 0.0, FiniteDifferenceMethod::Central) {
        Ok(_) => return Err(anyhow!("invalid step size unexpectedly succeeded")),
        Err(error) => error,
    };

    assert!(error.to_string().contains("step size"));
    Ok(())
}
