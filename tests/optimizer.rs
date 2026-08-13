use anyhow::{anyhow, Result as TestResult};
use faer::Mat;
use lmopt::{Error, JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result, TerminationReason};

struct Quadratic;

impl LeastSquaresProblem<f64> for Quadratic {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        Ok(Mat::from_fn(1, 1, |_, _| parameters[(0, 0)] - 2.0))
    }

    fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
        Some(Mat::ones(1, 1))
    }
}

struct Rosenbrock;

impl LeastSquaresProblem<f64> for Rosenbrock {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];
        Ok(Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 - x } else { 10.0 * (y - x * x) }))
    }

    fn jacobian(&self, parameters: &Mat<f64>) -> Option<Mat<f64>> {
        let x = parameters[(0, 0)];
        Some(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => -1.0,
            (0, 1) => 0.0,
            (1, 0) => -20.0 * x,
            (1, 1) => 10.0,
            _ => unreachable!(),
        }))
    }
}

#[test]
fn quadratic_converges_beyond_the_first_accepted_step() -> TestResult<()> {
    let optimizer = LevenbergMarquardt::new().with_max_iterations(100).with_jacobian_method(JacobianMethod::UserProvided);

    let report = optimizer.minimize(&Quadratic, &Mat::zeros(1, 1))?;

    assert!(report.success, "{report:?}");
    assert!((report.solution_params[(0, 0)] - 2.0).abs() < 1e-10);
    assert!(report.objective_function < 1e-20);
    assert!(report.iterations > 1, "the first damped step is not exact");
    assert_eq!(report.iterations, report.accepted_steps + report.rejected_steps);
    assert!(report.residual_evaluations > report.iterations);
    Ok(())
}

#[test]
fn rosenbrock_reaches_the_minimum() -> TestResult<()> {
    let optimizer = LevenbergMarquardt::new()
        .with_max_iterations(250)
        .with_ftol(1e-12)
        .with_xtol(1e-12)
        .with_tau(1e-12)
        .with_jacobian_method(JacobianMethod::UserProvided);
    let initial = Mat::from_fn(2, 1, |i, _| if i == 0 { -1.2 } else { 1.0 });

    let report = optimizer.minimize(&Rosenbrock, &initial)?;

    assert!(report.success, "{report:?}");
    assert!((report.solution_params[(0, 0)] - 1.0).abs() < 1e-8);
    assert!((report.solution_params[(1, 0)] - 1.0).abs() < 1e-8);
    assert!(report.objective_function < 1e-16);
    assert_eq!(report.iterations, report.accepted_steps + report.rejected_steps);
    assert!(report.rejected_steps > 5, "benchmark scenario should exercise damping retries: {report:?}");
    Ok(())
}

#[test]
fn exact_initial_solution_is_successful() -> TestResult<()> {
    let optimizer = LevenbergMarquardt::new().with_jacobian_method(JacobianMethod::UserProvided);
    let initial = Mat::from_fn(1, 1, |_, _| 2.0);

    let report = optimizer.minimize(&Quadratic, &initial)?;

    assert!(report.success);
    assert_eq!(report.iterations, 0);
    assert_eq!(report.termination_reason, TerminationReason::Converged);
    Ok(())
}

struct NonFiniteResidual;

impl LeastSquaresProblem<f64> for NonFiniteResidual {
    fn residuals(&self, _parameters: &Mat<f64>) -> Result<Mat<f64>> {
        Ok(Mat::from_fn(1, 1, |_, _| f64::NAN))
    }
}

#[test]
fn non_finite_residual_is_rejected() -> TestResult<()> {
    let error = match LevenbergMarquardt::new().minimize(&NonFiniteResidual, &Mat::zeros(1, 1)) {
        Ok(_) => return Err(anyhow!("non-finite residual unexpectedly succeeded")),
        Err(error) => error,
    };

    assert!(error.to_string().contains("finite"), "{error}");
    Ok(())
}

#[test]
fn invalid_configuration_is_rejected() -> TestResult<()> {
    let error = match LevenbergMarquardt::new().with_tau(-1.0).minimize(&Quadratic, &Mat::zeros(1, 1)) {
        Ok(_) => return Err(anyhow!("invalid configuration unexpectedly succeeded")),
        Err(error) => error,
    };

    assert!(error.to_string().contains("tau"), "{error}");
    Ok(())
}

#[test]
fn default_auto_mode_uses_the_analytical_jacobian() -> TestResult<()> {
    let report = LevenbergMarquardt::new().minimize(&Quadratic, &Mat::zeros(1, 1))?;

    assert_eq!(report.jacobian_method_used, JacobianMethod::UserProvided);
    assert!(report.jacobian_evaluations > 0);
    Ok(())
}

struct QuadraticWithoutJacobian;

impl LeastSquaresProblem<f64> for QuadraticWithoutJacobian {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        Ok(Mat::from_fn(1, 1, |_, _| parameters[(0, 0)] - 2.0))
    }
}

#[test]
fn default_auto_mode_falls_back_to_central_differences() -> TestResult<()> {
    let report = LevenbergMarquardt::new().minimize(&QuadraticWithoutJacobian, &Mat::zeros(1, 1))?;

    assert!(report.success);
    assert_eq!(report.jacobian_method_used, JacobianMethod::NumericalCentral);
    Ok(())
}

#[test]
fn minimize_is_checked_while_optimize_preserves_a_partial_report() -> TestResult<()> {
    let optimizer = LevenbergMarquardt::new().with_max_iterations(1).with_jacobian_method(JacobianMethod::UserProvided);

    let report = optimizer.optimize(&Quadratic, &Mat::zeros(1, 1))?;
    assert!(!report.converged());
    assert_eq!(report.termination_reason, TerminationReason::MaxIterationsReached);

    let error = match optimizer.minimize(&Quadratic, &Mat::zeros(1, 1)) {
        Ok(_) => return Err(anyhow!("checked minimization unexpectedly converged")),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::NoConvergence {
            reason: TerminationReason::MaxIterationsReached,
            iterations: 1
        }
    ));
    Ok(())
}
