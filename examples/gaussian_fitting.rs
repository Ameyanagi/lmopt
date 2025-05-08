use anyhow::Context;
use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};

// Fitting a Gaussian curve: y = a * exp(-(x - b)² / (2 * c²)) + d
// Parameters: a (amplitude), b (center), c (width), d (offset)
#[derive(Debug)]
struct GaussianModel {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl LeastSquaresProblem<f64> for GaussianModel {
    fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
        let a = parameters[(0, 0)]; // amplitude
        let b = parameters[(1, 0)]; // center
        let c = parameters[(2, 0)]; // width
        let d = parameters[(3, 0)]; // offset

        let n = self.x_data.len();
        let mut residuals = faer::Mat::zeros(n, 1);

        for i in 0..n {
            let x = self.x_data[i];
            let y = self.y_data[i];

            let exponent = -((x - b) * (x - b)) / (2.0 * c * c);
            let predicted = a * exponent.exp() + d;

            // residual = predicted - observed
            residuals[(i, 0)] = predicted - y;
        }

        Ok(residuals)
    }

    fn jacobian(&self, parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
        let a = parameters[(0, 0)]; // amplitude
        let b = parameters[(1, 0)]; // center
        let c = parameters[(2, 0)]; // width
        let _d = parameters[(3, 0)]; // offset

        let n = self.x_data.len();
        let mut jacobian = faer::Mat::zeros(n, 4);

        for i in 0..n {
            let x = self.x_data[i];

            let exponent = -((x - b) * (x - b)) / (2.0 * c * c);
            let gaussian = exponent.exp();

            // Derivatives with respect to each parameter
            // ∂r/∂a = exp(-(x-b)²/(2c²))
            jacobian[(i, 0)] = gaussian;

            // ∂r/∂b = a * exp(-(x-b)²/(2c²)) * (x-b)/c²
            jacobian[(i, 1)] = a * gaussian * (x - b) / (c * c);

            // ∂r/∂c = a * exp(-(x-b)²/(2c²)) * (x-b)²/(c³)
            jacobian[(i, 2)] = a * gaussian * (x - b) * (x - b) / (c * c * c);

            // ∂r/∂d = 1
            jacobian[(i, 3)] = 1.0;
        }

        Some(jacobian)
    }
}

fn main() -> Result<()> {
    // Generate synthetic data with noise
    // Gaussian with a=2.0, b=5.0, c=1.5, d=0.5
    let x_data: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect(); // x from 0 to 10
    let mut y_data = Vec::with_capacity(x_data.len());

    // Parameters for the true Gaussian
    let true_a = 2.0; // amplitude
    let true_b = 5.0; // center
    let true_c = 1.5; // width
    let true_d = 0.5; // offset

    // Generate y values with some noise
    for &x in &x_data {
        let exponent = -((x - true_b) * (x - true_b)) / (2.0 * true_c * true_c);
        let true_y = true_a * exponent.exp() + true_d;

        // Add random noise (simple pseudo-random using a deterministic formula)
        let noise = 0.1 * ((x * 100.0).sin() * 0.1);
        y_data.push(true_y + noise);
    }

    // Create problem instance
    let problem = GaussianModel { x_data, y_data };

    // Initial guess - starting far from the true values
    let initial_guess = faer::Mat::from_fn(4, 1, |i, _| {
        match i {
            0 => 1.0, // a
            1 => 4.0, // b
            2 => 1.0, // c
            3 => 0.0, // d
            _ => unreachable!(),
        }
    });

    // Create optimizer
    let optimizer = LevenbergMarquardt::new()
        .with_max_iterations(100)
        .with_epsilon_1(1e-8)
        .with_epsilon_2(1e-8)
        .with_jacobian_method(JacobianMethod::UserProvided);

    // Solve the optimization problem
    let result = optimizer.minimize(&problem, &initial_guess).with_context(|| "Failed to fit Gaussian model")?;

    // Print the results
    println!("Gaussian Fitting Results");
    println!("========================");
    println!("Optimization complete after {} iterations", result.iterations);
    println!("Success: {}", result.success);
    println!("Termination reason: {:?}", result.termination_reason);
    println!("Execution time: {:?}", result.execution_time);

    // Extract the optimized parameters
    let fitted_a = result.solution_params[(0, 0)];
    let fitted_b = result.solution_params[(1, 0)];
    let fitted_c = result.solution_params[(2, 0)];
    let fitted_d = result.solution_params[(3, 0)];

    println!("\nFitted Gaussian: f(x) = {:.4} * exp(-((x - {:.4})² / (2 * {:.4}²))) + {:.4}", fitted_a, fitted_b, fitted_c, fitted_d);
    println!("Final sum of squares: {:.6e}", result.objective_function);

    // Compare with the true values
    println!("\nTrue values: a={:.4}, b={:.4}, c={:.4}, d={:.4}", true_a, true_b, true_c, true_d);
    println!("Errors:");
    println!("  Amplitude (a): {:.6}", (fitted_a - true_a).abs());
    println!("  Center (b): {:.6}", (fitted_b - true_b).abs());
    println!("  Width (c): {:.6}", (fitted_c - true_c).abs());
    println!("  Offset (d): {:.6}", (fitted_d - true_d).abs());

    // Optionally, print the data points and fitted curve
    println!("\nData points and fitted curve (first 10 points):");
    println!("   x      y_data    y_fitted");
    for i in 0..10.min(problem.x_data.len()) {
        let x = problem.x_data[i];
        let y = problem.y_data[i];

        let exponent = -((x - fitted_b) * (x - fitted_b)) / (2.0 * fitted_c * fitted_c);
        let y_fitted = fitted_a * exponent.exp() + fitted_d;

        println!("{:6.2} {:9.4} {:9.4}", x, y, y_fitted);
    }

    Ok(())
}
