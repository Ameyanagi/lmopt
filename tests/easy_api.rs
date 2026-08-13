use anyhow::{anyhow, Result as TestResult};
use faer::{Col, Mat};
use lmopt::{least_squares, Error, LeastSquaresProblem, LevenbergMarquardt, Result};

#[test]
fn closure_api_fits_without_exposing_matrix_types() -> TestResult<()> {
    let xs = [0.0, 1.0, 2.0, 3.0];
    let ys = [1.0, 3.0, 5.0, 7.0];

    let report = least_squares(&[0.0, 0.0], |parameters| {
        let [slope, intercept] = parameters else { return Vec::new() };
        xs.iter().zip(ys).map(|(x, y)| slope * x + intercept - y).collect()
    })?;

    assert!((report.parameters()[0] - 2.0).abs() < 1e-6);
    assert!((report.parameters()[1] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn configured_fallible_closure_propagates_model_errors() -> TestResult<()> {
    let error = match LevenbergMarquardt::new().try_minimize_fn(&[0.0], |_| Err(anyhow!("model unavailable"))) {
        Ok(_) => return Err(anyhow!("fallible model unexpectedly succeeded")),
        Err(error) => error,
    };

    assert!(matches!(error, Error::UserFunction(message) if message == "model unavailable"));
    Ok(())
}

struct ScalarProblem;

impl LeastSquaresProblem<f64> for ScalarProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        Ok(Mat::from_fn(1, 1, |_, _| parameters[(0, 0)] - 3.0))
    }
}

#[test]
fn advanced_api_accepts_a_faer_column_as_the_initial_guess() -> TestResult<()> {
    let initial_guess = Col::from_fn(1, |_| 0.0);
    let report = LevenbergMarquardt::new().minimize(&ScalarProblem, &initial_guess)?;

    assert!((report.parameters()[0] - 3.0).abs() < 1e-6);
    Ok(())
}
