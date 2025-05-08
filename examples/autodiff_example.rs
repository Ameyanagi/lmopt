//! Example demonstrating the usage of automatic differentiation for Jacobian calculation.
//!
//! This example requires the 'autodiff' feature and nightly Rust.
//! Run with:
//!    cargo +nightly run --example autodiff_example --features autodiff

// Main implementation when autodiff feature is enabled
#[cfg(feature = "autodiff")]
mod impl_autodiff {
    use anyhow::Context;
    use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};
    use std::time::Instant;

    /// Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    /// This is a common test function for optimization algorithms.
    /// Minimum at (1, 1) with a function value of 0.
    pub struct RosenbrockProblem;

    impl LeastSquaresProblem<f64> for RosenbrockProblem {
        fn residuals(&self, parameters: &faer::Mat<f64>) -> Result<faer::Mat<f64>> {
            let x = parameters[(0, 0)];
            let y = parameters[(1, 0)];

            // Residuals: [(1-x), 10(y-x²)]
            let mut residuals = faer::Mat::zeros(2, 1);
            residuals[(0, 0)] = 1.0 - x;
            residuals[(1, 0)] = 10.0 * (y - x * x);

            Ok(residuals)
        }

        // We deliberately don't provide an analytical Jacobian
        // to demonstrate autodiff capabilities
    }

    /// A more complex example: Fitting a Gaussian function
    /// y = a * exp(-(x - b)² / (2 * c²)) + d
    pub struct GaussianModel {
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

        // No analytical Jacobian provided - autodiff will be used instead
    }

    pub fn run_example() -> Result<()> {
        println!("AutoDiff Jacobian Example");
        println!("=========================\n");

        // Example 1: Rosenbrock function
        println!("Example 1: Rosenbrock Function");
        println!("-----------------------------");

        let problem = RosenbrockProblem;
        let initial_guess = faer::Mat::from_fn(2, 1, |i, _| if i == 0 { -1.2 } else { 1.0 });

        // Create optimizer using autodiff
        let optimizer = LevenbergMarquardt::new()
            .with_max_iterations(100)
            .with_epsilon_1(1e-10)
            .with_epsilon_2(1e-10)
            .with_jacobian_method(JacobianMethod::AutoDiff);

        // Compare with numerical differentiation
        let numerical_optimizer = LevenbergMarquardt::new()
            .with_max_iterations(100)
            .with_epsilon_1(1e-10)
            .with_epsilon_2(1e-10)
            .with_jacobian_method(JacobianMethod::NumericalCentral);

        // Solve with autodiff
        let start = Instant::now();
        let result = optimizer.minimize(&problem, &initial_guess).with_context(|| "Failed to minimize Rosenbrock function with autodiff")?;
        let autodiff_time = start.elapsed();

        println!("AutoDiff Optimization Results:");
        println!("  Solution: ({:.6}, {:.6})", result.solution_params[(0, 0)], result.solution_params[(1, 0)]);
        println!(
            "  Error from true minimum: {:.6e}",
            ((result.solution_params[(0, 0)] - 1.0).powi(2) + (result.solution_params[(1, 0)] - 1.0).powi(2)).sqrt()
        );
        println!("  Iterations: {}", result.iterations);
        println!("  Final objective value: {:.6e}", result.objective_function);
        println!("  Execution time: {:?}", autodiff_time);

        // Solve with numerical differentiation
        let start = Instant::now();
        let num_result = numerical_optimizer
            .minimize(&problem, &initial_guess)
            .with_context(|| "Failed to minimize Rosenbrock function with numerical differentiation")?;
        let numerical_time = start.elapsed();

        println!("\nNumerical Differentiation Results:");
        println!("  Solution: ({:.6}, {:.6})", num_result.solution_params[(0, 0)], num_result.solution_params[(1, 0)]);
        println!(
            "  Error from true minimum: {:.6e}",
            ((num_result.solution_params[(0, 0)] - 1.0).powi(2) + (num_result.solution_params[(1, 0)] - 1.0).powi(2)).sqrt()
        );
        println!("  Iterations: {}", num_result.iterations);
        println!("  Final objective value: {:.6e}", num_result.objective_function);
        println!("  Execution time: {:?}", numerical_time);

        println!("\nPerformance Comparison:");
        if autodiff_time < numerical_time {
            println!("  AutoDiff was {:.1}x faster!", numerical_time.as_secs_f64() / autodiff_time.as_secs_f64());
        } else {
            println!("  Numerical differentiation was {:.1}x faster!", autodiff_time.as_secs_f64() / numerical_time.as_secs_f64());
        }

        // Example 2: Gaussian fitting
        println!("\n\nExample 2: Gaussian Function Fitting");
        println!("----------------------------------");

        // Generate synthetic data with noise
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

        let gaussian_problem = GaussianModel { x_data, y_data };
        let initial_guess = faer::Mat::from_fn(4, 1, |i, _| {
            match i {
                0 => 1.0, // a
                1 => 4.0, // b
                2 => 1.0, // c
                3 => 0.0, // d
                _ => unreachable!(),
            }
        });

        // Solve with autodiff
        let start = Instant::now();
        let result = optimizer.minimize(&gaussian_problem, &initial_guess).with_context(|| "Failed to fit Gaussian with autodiff")?;
        let autodiff_time = start.elapsed();

        // Extract the optimized parameters
        let fitted_a = result.solution_params[(0, 0)];
        let fitted_b = result.solution_params[(1, 0)];
        let fitted_c = result.solution_params[(2, 0)];
        let fitted_d = result.solution_params[(3, 0)];

        println!("AutoDiff Gaussian Fitting Results:");
        println!("  Fitted parameters: a={:.4}, b={:.4}, c={:.4}, d={:.4}", fitted_a, fitted_b, fitted_c, fitted_d);
        println!("  True parameters:   a={:.4}, b={:.4}, c={:.4}, d={:.4}", true_a, true_b, true_c, true_d);
        println!("  Iterations: {}", result.iterations);
        println!("  Final objective value: {:.6e}", result.objective_function);
        println!("  Execution time: {:?}", autodiff_time);

        println!("\nParameter Errors:");
        println!("  Amplitude (a): {:.6}", (fitted_a - true_a).abs());
        println!("  Center (b): {:.6}", (fitted_b - true_b).abs());
        println!("  Width (c): {:.6}", (fitted_c - true_c).abs());
        println!("  Offset (d): {:.6}", (fitted_d - true_d).abs());

        // Solve with numerical differentiation
        let start = Instant::now();
        let num_result = numerical_optimizer
            .minimize(&gaussian_problem, &initial_guess)
            .with_context(|| "Failed to fit Gaussian with numerical differentiation")?;
        let numerical_time = start.elapsed();

        // Extract the optimized parameters
        let num_a = num_result.solution_params[(0, 0)];
        let num_b = num_result.solution_params[(1, 0)];
        let num_c = num_result.solution_params[(2, 0)];
        let num_d = num_result.solution_params[(3, 0)];

        println!("\nNumerical Differentiation Results:");
        println!("  Fitted parameters: a={:.4}, b={:.4}, c={:.4}, d={:.4}", num_a, num_b, num_c, num_d);
        println!("  Iterations: {}", num_result.iterations);
        println!("  Final objective value: {:.6e}", num_result.objective_function);
        println!("  Execution time: {:?}", numerical_time);

        println!("\nPerformance Comparison:");
        if autodiff_time < numerical_time {
            println!("  AutoDiff was {:.1}x faster!", numerical_time.as_secs_f64() / autodiff_time.as_secs_f64());
        } else {
            println!("  Numerical differentiation was {:.1}x faster!", autodiff_time.as_secs_f64() / numerical_time.as_secs_f64());
        }

        // Accuracy comparison
        println!("\nAccuracy Comparison (Parameter Error Ratios):");
        println!("  Amplitude: {:.2}", (num_a - true_a).abs() / (fitted_a - true_a).abs());
        println!("  Center: {:.2}", (num_b - true_b).abs() / (fitted_b - true_b).abs());
        println!("  Width: {:.2}", (num_c - true_c).abs() / (fitted_c - true_c).abs());
        println!("  Offset: {:.2}", (num_d - true_d).abs() / (fitted_d - true_d).abs());

        Ok(())
    }
}

// Main function dispatch based on features
#[cfg(feature = "autodiff")]
fn main() -> lmopt::Result<()> {
    impl_autodiff::run_example()
}

// Fallback main function when autodiff feature is not enabled
#[cfg(not(feature = "autodiff"))]
fn main() {
    println!("This example requires the 'autodiff' feature and nightly Rust.");
    println!("Run with: cargo +nightly run --example autodiff_example --features autodiff");
}
