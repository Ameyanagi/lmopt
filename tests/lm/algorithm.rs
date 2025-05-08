use faer::Mat;
use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};

// Simple quadratic problem: f(x) = (x - 2)²
// Minimum at x = 2
struct SimpleQuadratic;

impl LeastSquaresProblem<f64> for SimpleQuadratic {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];

        // Residual: [x - 2]
        let mut residuals = Mat::zeros(1, 1);
        residuals[(0, 0)] = x - 2.0;

        Ok(residuals)
    }

    fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
        // Jacobian: [1]
        let mut jacobian = Mat::zeros(1, 1);
        jacobian[(0, 0)] = 1.0;

        Some(jacobian)
    }
}

#[test]
#[ignore] // Ignore until the LM algorithm is implemented
fn test_simple_quadratic_optimization() {
    let problem = SimpleQuadratic;
    let initial_guess = Mat::from_fn(1, 1, |_, _| 0.0); // Start at x = 0

    let optimizer = LevenbergMarquardt::new().with_max_iterations(100).with_jacobian_method(JacobianMethod::UserProvided);

    let result = optimizer.minimize(&problem, &initial_guess).unwrap();

    // Check that we found the minimum at x = 2
    assert!(result.success);
    assert!((result.solution_params[(0, 0)] - 2.0).abs() < 1e-8);
    assert!(result.objective_function < 1e-16);
}

// Linear model: y = a*x + b
struct LinearModel {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl LeastSquaresProblem<f64> for LinearModel {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let a = parameters[(0, 0)];
        let b = parameters[(1, 0)];

        let n = self.x_data.len();
        let mut residuals = Mat::zeros(n, 1);

        for i in 0..n {
            let x = self.x_data[i];
            let y = self.y_data[i];
            let predicted = a * x + b;
            // residual = predicted - observed
            residuals[(i, 0)] = predicted - y;
        }

        Ok(residuals)
    }

    fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
        let n = self.x_data.len();
        let mut jacobian = Mat::zeros(n, 2);

        for i in 0..n {
            let x = self.x_data[i];
            // ∂r_i/∂a = x_i
            jacobian[(i, 0)] = x;
            // ∂r_i/∂b = 1
            jacobian[(i, 1)] = 1.0;
        }

        Some(jacobian)
    }
}

#[test]
#[ignore] // Ignore until the LM algorithm is implemented
fn test_linear_model_fitting() {
    // Generate some sample data: y = 2x + 3 with noise
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![3.1, 5.2, 6.9, 9.1, 11.0, 13.1]; // 2*x + 3 with some noise

    let problem = LinearModel { x_data, y_data };

    // Initial guess: a = 1, b = 1
    let initial_guess = Mat::from_fn(2, 1, |_, _| 1.0);

    let optimizer = LevenbergMarquardt::new().with_max_iterations(100).with_jacobian_method(JacobianMethod::UserProvided);

    let result = optimizer.minimize(&problem, &initial_guess).unwrap();

    // Check that we found parameters close to a = 2, b = 3
    assert!(result.success);
    assert!((result.solution_params[(0, 0)] - 2.0).abs() < 0.1);
    assert!((result.solution_params[(1, 0)] - 3.0).abs() < 0.1);
}
