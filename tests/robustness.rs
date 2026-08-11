use faer::Mat;
use lmopt::{Error, JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};
use proptest::prelude::*;

struct LinearData {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl LeastSquaresProblem<f64> for LinearData {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let slope = parameters[(0, 0)];
        let intercept = parameters[(1, 0)];
        Ok(Mat::from_fn(self.xs.len(), 1, |i, _| slope * self.xs[i] + intercept - self.ys[i]))
    }

    fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
        Some(Mat::from_fn(self.xs.len(), 2, |i, j| if j == 0 { self.xs[i] } else { 1.0 }))
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn exact_linear_models_converge_from_varied_initial_guesses(
        slope in -20.0f64..20.0,
        intercept in -20.0f64..20.0,
        initial_slope in -10.0f64..10.0,
        initial_intercept in -10.0f64..10.0,
    ) {
        let xs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let ys = xs.iter().map(|x| slope * x + intercept).collect();
        let problem = LinearData { xs, ys };
        let initial = Mat::from_fn(2, 1, |i, _| if i == 0 { initial_slope } else { initial_intercept });
        let report = LevenbergMarquardt::new()
            .with_jacobian_method(JacobianMethod::UserProvided)
            .minimize(&problem, &initial)
            .unwrap();

        prop_assert!(report.success, "{report:?}");
        prop_assert!((report.solution_params[(0, 0)] - slope).abs() < 1e-7);
        prop_assert!((report.solution_params[(1, 0)] - intercept).abs() < 1e-7);
        prop_assert!(report.objective_function < 1e-12);
    }
}

#[test]
fn noisy_linear_data_converges_to_the_least_squares_fit() {
    let xs = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let noise = [0.08, -0.04, 0.02, -0.06, 0.01, 0.05, -0.03];
    let ys = xs.iter().zip(noise).map(|(x, error)| 2.5 * x - 0.75 + error).collect();
    let problem = LinearData { xs, ys };

    let report = LevenbergMarquardt::new()
        .with_jacobian_method(JacobianMethod::UserProvided)
        .minimize(&problem, &Mat::zeros(2, 1))
        .unwrap();

    assert!(report.success, "{report:?}");
    assert!((report.solution_params[(0, 0)] - 2.5).abs() < 0.02);
    assert!((report.solution_params[(1, 0)] + 0.75).abs() < 0.02);
    assert!(report.objective_function < 0.02);
}

struct Underdetermined;

impl LeastSquaresProblem<f64> for Underdetermined {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        Ok(Mat::from_fn(1, 1, |_, _| parameters[(0, 0)] + parameters[(1, 0)] - 2.0))
    }

    fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
        Some(Mat::ones(1, 2))
    }
}

#[test]
fn underdetermined_problem_reaches_the_symmetric_minimum_norm_solution() {
    let report = LevenbergMarquardt::new()
        .with_jacobian_method(JacobianMethod::UserProvided)
        .minimize(&Underdetermined, &Mat::zeros(2, 1))
        .unwrap();

    assert!(report.success, "{report:?}");
    assert!((report.solution_params[(0, 0)] - 1.0).abs() < 1e-8);
    assert!((report.solution_params[(1, 0)] - 1.0).abs() < 1e-8);
    assert!(report.objective_function < 1e-16);
}

struct SquareRootDomain;

impl LeastSquaresProblem<f64> for SquareRootDomain {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        if x < 0.0 {
            return Err(Error::UserFunction("sqrt parameter must be non-negative".to_string()));
        }
        Ok(Mat::from_fn(1, 1, |_, _| x.sqrt() - 2.0))
    }
}

#[test]
fn forward_difference_supports_a_one_sided_parameter_domain() {
    let initial = Mat::from_fn(1, 1, |_, _| 0.0);
    let report = LevenbergMarquardt::new()
        .with_max_iterations(200)
        .with_jacobian_method(JacobianMethod::NumericalForward)
        .minimize(&SquareRootDomain, &initial)
        .unwrap();

    assert!(report.success, "{report:?}");
    assert!((report.solution_params[(0, 0)] - 4.0).abs() < 1e-6);
}
