use anyhow::Context;
use lmopt::{JacobianMethod, LeastSquaresProblem, LevenbergMarquardt, Result};
use std::time::Instant;

// Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
// Minimum at (1, 1)
#[derive(Debug)]
struct RosenbrockProblem;

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

    fn jacobian(&self, parameters: &faer::Mat<f64>) -> Option<faer::Mat<f64>> {
        let x = parameters[(0, 0)];

        // Jacobian:
        // [-1, 0]
        // [-20x, 10]
        let mut jacobian = faer::Mat::zeros(2, 2);
        jacobian[(0, 0)] = -1.0;
        jacobian[(0, 1)] = 0.0;
        jacobian[(1, 0)] = -20.0 * x;
        jacobian[(1, 1)] = 10.0;

        Some(jacobian)
    }
}

fn main() -> Result<()> {
    // Compare different Jacobian calculation methods on the Rosenbrock function

    // Initial guess far from the minimum
    let initial_guess = faer::Mat::from_fn(2, 1, |_i, _| -1.0);
    let problem = RosenbrockProblem;

    println!("Comparing different Jacobian calculation methods on the Rosenbrock function");
    println!("Initial guess: ({}, {})", initial_guess[(0, 0)], initial_guess[(1, 0)]);
    println!("True minimum: (1, 1)\n");

    // Test all available Jacobian methods
    for method in [
        JacobianMethod::UserProvided,
        JacobianMethod::NumericalForward,
        JacobianMethod::NumericalCentral,
        JacobianMethod::NumericalBackward,
        JacobianMethod::AutoDiff, // Will typically fall back to numerical methods
    ] {
        let start = Instant::now();

        // Create optimizer with specific Jacobian method
        let optimizer = LevenbergMarquardt::new()
            .with_max_iterations(100)
            .with_epsilon_1(1e-10)
            .with_epsilon_2(1e-10)
            .with_jacobian_method(method);

        // Solve the optimization problem
        let result = optimizer.minimize(&problem, &initial_guess).with_context(|| format!("Failed to minimize with {:?}", method))?;

        let duration = start.elapsed();

        // Extract the optimized parameters
        let x = result.solution_params[(0, 0)];
        let y = result.solution_params[(1, 0)];

        // Print results for this method
        println!("{:?}:", method);
        println!("  Solution: ({:.6}, {:.6})", x, y);
        println!("  Error from true minimum: {:.6e}", ((x - 1.0).powi(2) + (y - 1.0).powi(2)).sqrt());
        println!("  Iterations: {}", result.iterations);
        println!("  Final objective value: {:.6e}", result.objective_function);
        println!("  Computation time: {:?}", duration);
        println!("  Execution time (reported): {:?}", result.execution_time);
        println!("  Success: {}", result.success);
        println!("  Termination reason: {:?}", result.termination_reason);
        println!();
    }

    Ok(())
}
