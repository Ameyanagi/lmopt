use anyhow::Context;
use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};

// A simple problem fitting a line: y = a*x + b
// This demonstrates the basic usage of the lmopt library
#[derive(Debug)]
struct LinearModel {
    // X values for the data points
    x_data: Vec<f64>,
    // Y values for the data points
    y_data: Vec<f64>,
}

impl LeastSquaresProblem<f64> for LinearModel {
    fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
        let a = parameters[(0, 0)];
        let b = parameters[(1, 0)];

        let n = self.x_data.len();
        let mut residuals = faer::Mat::zeros(n, 1);

        for i in 0..n {
            let x = self.x_data[i];
            let y = self.y_data[i];
            let predicted = a * x + b;
            // residual = predicted - observed
            residuals[(i, 0)] = predicted - y;
        }

        Ok(residuals)
    }

    fn jacobian(&self, _parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
        let n = self.x_data.len();
        let mut jacobian = faer::Mat::zeros(n, 2);

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

fn main() -> Result<()> {
    // Generate some sample data: y = 2x + 3 with noise
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![3.1, 5.2, 6.9, 9.1, 11.0, 13.1]; // 2*x + 3 with some noise

    let problem = LinearModel { x_data, y_data };

    // Initial guess
    let initial_guess = faer::Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 1.0 }); // a=1, b=1

    // Create optimizer with fluent API
    let optimizer = LevenbergMarquardt::new()
        .with_max_iterations(100)
        .with_epsilon_1(1e-8)
        .with_epsilon_2(1e-8)
        .with_jacobian_method(JacobianMethod::UserProvided);

    // Solve the optimization problem
    let result = optimizer.minimize(&problem, &initial_guess).with_context(|| "Failed to minimize the linear model")?;

    // Print the results
    println!("Optimization complete after {} iterations", result.iterations);
    println!("Success: {}", result.success);
    println!("Termination reason: {:?}", result.termination_reason);

    // Extract the optimized parameters
    let a = result.solution_params[(0, 0)];
    let b = result.solution_params[(1, 0)];

    println!("\nFitted line: y = {:.4}*x + {:.4}", a, b);
    println!("Final sum of squares: {:.6e}", result.objective_function);
    println!("Execution time: {:?}", result.execution_time);

    // Compare with the true values (a=2, b=3)
    println!("\nTrue values: a=2, b=3");
    println!("Error in a: {:.6}", (a - 2.0).abs());
    println!("Error in b: {:.6}", (b - 3.0).abs());

    Ok(())
}
