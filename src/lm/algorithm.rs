use super::{
    convergence::check_convergence,
    trust_region::{adjust_lambda, calculate_parameter_update},
};
use crate::lm::{LevenbergMarquardt, MinimizationReport, TerminationReason};
use crate::{utils::jacobian::get_jacobian_calculator, Error, LeastSquaresProblem, Result};
use faer::mat::Mat;
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::ops::AddAssign;
use std::time::Instant;

// Calculate ||residuals||^2 / 2 which is the objective function
fn calculate_objective_function<T>(residuals: &Mat<T>) -> T
where
    T: RealField + Copy + Float + FromPrimitive,
{
    let mut sum = T::zero();
    for i in 0..residuals.nrows() {
        sum = sum + residuals[(i, 0)] * residuals[(i, 0)];
    }
    sum * T::from_f64(0.5).unwrap()
}

// Calculate the norm of a vector
fn vector_norm<T>(vec: &Mat<T>) -> T
where
    T: RealField + Copy + Float,
{
    // Calculate Euclidean norm manually
    let mut sum = T::zero();
    for i in 0..vec.nrows() {
        sum = sum + vec[(i, 0)] * vec[(i, 0)];
    }
    sum.sqrt()
}

impl LevenbergMarquardt {
    /// Core implementation of the Levenberg-Marquardt algorithm
    pub(crate) fn minimize_impl<T, P>(&self, problem: &P, initial_guess: &Mat<T>) -> Result<MinimizationReport<T>>
    where
        T: RealField + Copy + Float + FromPrimitive + AddAssign + 'static,
        P: LeastSquaresProblem<T>,
    {
        let start_time = Instant::now();

        // Validate initial guess dimensions
        if initial_guess.ncols() != 1 {
            return Err(Error::DimensionMismatch(format!("Initial guess must be a column vector, got dimensions {}x{}", initial_guess.nrows(), initial_guess.ncols())).into());
        }

        // Set up variables
        let mut params = initial_guess.clone();
        let n_params = params.nrows();

        // Compute initial residuals
        let mut residuals = problem.residuals(&params)?;
        let m_residuals = residuals.nrows();

        if residuals.ncols() != 1 {
            return Err(Error::DimensionMismatch(format!("Residuals must be a column vector, got dimensions {}x{}", residuals.nrows(), residuals.ncols())).into());
        }

        // Calculate initial objective function value
        let mut objective_function = calculate_objective_function(&residuals);

        // Initialize the scaling diagonal matrix
        let mut diag = Mat::ones(n_params, 1);

        // Calculate initial jacobian
        let calculator = get_jacobian_calculator(self.jacobian_method, self.numerical_diff_step_size);
        let mut jacobian = calculator.calculate_jacobian(problem, &params)?;

        if jacobian.nrows() != m_residuals || jacobian.ncols() != n_params {
            return Err(Error::DimensionMismatch(format!(
                "Jacobian dimensions ({}x{}) don't match expected ({}x{})",
                jacobian.nrows(),
                jacobian.ncols(),
                m_residuals,
                n_params
            ))
            .into());
        }

        // Scale the diagonal based on the Jacobian if needed
        // Use the column norms of the Jacobian to scale the parameters
        for j in 0..n_params {
            let mut col_norm = T::zero();
            for i in 0..m_residuals {
                col_norm += jacobian[(i, j)] * jacobian[(i, j)];
            }
            col_norm = col_norm.sqrt();

            if col_norm > T::zero() {
                diag[(j, 0)] = col_norm;
            }
        }

        // Initialize damping parameter (lambda)
        let tau = T::from_f64(self.tau).unwrap();

        // Find max value in diagonal manually
        let mut max_diag = T::zero();
        for i in 0..diag.nrows() {
            if diag[(i, 0)] > max_diag {
                max_diag = diag[(i, 0)];
            }
        }

        let mut lambda = tau * max_diag;

        // Main optimization loop
        let mut iterations = 0;
        let max_iterations = self.max_iterations;
        let epsilon_1 = T::from_f64(self.epsilon_1).unwrap();
        let epsilon_2 = T::from_f64(self.epsilon_2).unwrap();

        let mut residuals_norm = vector_norm(&residuals);

        // Initialize success flag for termination
        let mut success = false;
        let mut termination_reason = TerminationReason::MaxIterationsReached;

        while iterations < max_iterations {
            iterations += 1;

            // Solve the trust region subproblem
            let update = calculate_parameter_update(&jacobian, &residuals, lambda, &diag)?;

            // Check if the predicted reduction is too small
            if update.predicted_reduction <= T::zero() {
                termination_reason = TerminationReason::Other("No predicted reduction possible".to_string());
                break;
            }

            // Apply the parameter update
            let mut new_params = params.clone();
            for i in 0..n_params {
                new_params[(i, 0)] += update.step[(i, 0)];
            }

            // Compute new residuals
            let new_residuals = problem.residuals(&new_params)?;
            let new_residuals_norm = vector_norm(&new_residuals);

            // Compute actual reduction
            let new_objective = calculate_objective_function(&new_residuals);
            let actual_reduction = objective_function - new_objective;

            // Calculate ratio of actual to predicted reduction
            let ratio = if update.predicted_reduction.abs() < T::epsilon() {
                T::zero()
            } else {
                actual_reduction / update.predicted_reduction
            };

            // Check if this step was successful (we reduced the objective function)
            let step_success = actual_reduction > T::zero();

            // Adjust lambda based on the success of the iteration
            lambda = adjust_lambda(lambda, ratio, step_success);

            // If this was a successful step, update the current solution
            if step_success {
                params = new_params;
                residuals = new_residuals;
                objective_function = new_objective;
                residuals_norm = new_residuals_norm;

                // Recompute the Jacobian at the new point
                jacobian = calculator.calculate_jacobian(problem, &params)?;
            }

            // Calculate norms and check convergence
            let params_norm = vector_norm(&params);
            let step_norm = update.step_norm;

            // Check if we should terminate based on convergence criteria
            if let Some(reason) = check_convergence(residuals_norm, new_residuals_norm, step_norm, params_norm, epsilon_1, epsilon_2) {
                termination_reason = reason;
                success = true;
                break;
            }

            // Check for very small residuals (solution found)
            let residual_tolerance = T::from_f64(1e-14).unwrap();
            if residuals_norm < residual_tolerance {
                termination_reason = TerminationReason::Converged;
                success = true;
                break;
            }
        }

        // Record execution time
        let execution_time = start_time.elapsed();

        // Return the final result
        Ok(MinimizationReport {
            solution_params: params,
            residuals,
            objective_function,
            iterations,
            jacobian: Some(jacobian),
            jacobian_method_used: calculator.method_used(),
            success,
            termination_reason,
            execution_time,
        })
    }
}
