use super::{
    convergence::check_convergence,
    trust_region::{adjust_lambda, calculate_parameter_update, NormalEquations},
};
use crate::lm::{LevenbergMarquardt, MinimizationReport, TerminationReason};
use crate::{utils::jacobian::get_jacobian_calculator, Error, LeastSquaresProblem, Result};
use faer::mat::Mat;
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::cell::Cell;
use std::ops::AddAssign;
use std::time::Instant;

struct CountingProblem<'a, P> {
    inner: &'a P,
    residual_evaluations: Cell<usize>,
    jacobian_evaluations: Cell<usize>,
}

impl<'a, P> CountingProblem<'a, P> {
    fn new(inner: &'a P) -> Self {
        Self {
            inner,
            residual_evaluations: Cell::new(0),
            jacobian_evaluations: Cell::new(0),
        }
    }
}

impl<T, P> LeastSquaresProblem<T> for CountingProblem<'_, P>
where
    T: RealField + Copy,
    P: LeastSquaresProblem<T>,
{
    fn residuals(&self, parameters: &Mat<T>) -> Result<Mat<T>> {
        self.residual_evaluations.set(self.residual_evaluations.get() + 1);
        self.inner.residuals(parameters)
    }

    fn try_jacobian(&self, parameters: &Mat<T>) -> Result<Option<Mat<T>>> {
        self.jacobian_evaluations.set(self.jacobian_evaluations.get() + 1);
        self.inner.try_jacobian(parameters)
    }
}

fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::InvalidParameter(format!("{name} must be finite and greater than zero, got {value}")));
    }
    Ok(())
}

fn validate_vector<T>(vector: &Mat<T>, name: &str, expected_rows: Option<usize>) -> Result<()>
where
    T: RealField + Copy + Float,
{
    if vector.ncols() != 1 {
        return Err(Error::DimensionMismatch(format!(
            "{name} must be a column vector, got dimensions {}x{}",
            vector.nrows(),
            vector.ncols()
        )));
    }
    if vector.nrows() == 0 {
        return Err(Error::DimensionMismatch(format!("{name} must not be empty")));
    }
    if let Some(expected_rows) = expected_rows {
        if vector.nrows() != expected_rows {
            return Err(Error::DimensionMismatch(format!("{name} has {} rows, expected {expected_rows}", vector.nrows())));
        }
    }
    for i in 0..vector.nrows() {
        if !Float::is_finite(vector[(i, 0)]) {
            return Err(Error::Numerical(format!("{name} must contain only finite values; element {i} is not finite")));
        }
    }
    Ok(())
}

fn validate_jacobian<T>(jacobian: &Mat<T>, n_residuals: usize, n_params: usize) -> Result<()>
where
    T: RealField + Copy + Float,
{
    if jacobian.nrows() != n_residuals || jacobian.ncols() != n_params {
        return Err(Error::DimensionMismatch(format!(
            "Jacobian dimensions ({}x{}) don't match expected ({}x{})",
            jacobian.nrows(),
            jacobian.ncols(),
            n_residuals,
            n_params
        )));
    }
    for j in 0..jacobian.ncols() {
        for i in 0..jacobian.nrows() {
            if !Float::is_finite(jacobian[(i, j)]) {
                return Err(Error::Numerical(format!("Jacobian must contain only finite values; element ({i}, {j}) is not finite")));
            }
        }
    }
    Ok(())
}

// Calculate ||residuals||^2 / 2 which is the objective function
fn calculate_objective_function<T>(residuals: &Mat<T>) -> T
where
    T: RealField + Copy + Float + FromPrimitive,
{
    let norm = residuals.norm_l2();
    norm * norm * T::from_f64(0.5).unwrap()
}

// Calculate the norm of a vector
fn vector_norm<T>(vec: &Mat<T>) -> T
where
    T: RealField + Copy + Float,
{
    vec.norm_l2()
}

impl LevenbergMarquardt {
    /// Core implementation of the Levenberg-Marquardt algorithm
    pub(crate) fn minimize_impl<T, P>(&self, problem: &P, initial_guess: &Mat<T>) -> Result<MinimizationReport<T>>
    where
        T: RealField + Copy + Float + FromPrimitive + AddAssign + 'static,
        P: LeastSquaresProblem<T>,
    {
        let start_time = Instant::now();

        if self.max_iterations == 0 {
            return Err(Error::InvalidParameter("max_iterations must be greater than zero".to_string()));
        }
        validate_positive_finite(self.epsilon_1, "epsilon_1")?;
        validate_positive_finite(self.epsilon_2, "epsilon_2")?;
        validate_positive_finite(self.tau, "tau")?;
        if matches!(
            self.jacobian_method,
            crate::JacobianMethod::Auto | crate::JacobianMethod::NumericalCentral | crate::JacobianMethod::NumericalForward | crate::JacobianMethod::NumericalBackward
        ) {
            validate_positive_finite(self.numerical_diff_step_size, "numerical_diff_step_size")?;
        }

        validate_vector(initial_guess, "Initial guess", None)?;
        let problem = CountingProblem::new(problem);

        // Set up variables
        let mut params = initial_guess.clone();
        let n_params = params.nrows();

        // Compute initial residuals
        let mut residuals = problem.residuals(&params)?;
        let m_residuals = residuals.nrows();
        validate_vector(&residuals, "Residuals", None)?;

        // Calculate initial objective function value
        let mut objective_function = calculate_objective_function(&residuals);
        if !Float::is_finite(objective_function) {
            return Err(Error::Numerical("Initial objective value is not finite".to_string()));
        }

        // Avoid trying to factor a singular system when the initial point is
        // already an exact solution.
        if objective_function == T::zero() {
            return Ok(MinimizationReport {
                solution_params: params,
                residuals,
                objective_function,
                iterations: 0,
                jacobian: None,
                jacobian_method_used: self.jacobian_method,
                success: true,
                termination_reason: TerminationReason::Converged,
                execution_time: start_time.elapsed(),
                residual_evaluations: problem.residual_evaluations.get(),
                jacobian_evaluations: problem.jacobian_evaluations.get(),
                accepted_steps: 0,
                rejected_steps: 0,
                final_lambda: T::zero(),
            });
        }

        // Initialize the scaling diagonal matrix
        let mut diag = Mat::ones(n_params, 1);

        // Calculate initial jacobian
        let calculator = get_jacobian_calculator(self.jacobian_method, self.numerical_diff_step_size);
        let mut jacobian = calculator.calculate_jacobian_with_residuals(&problem, &params, &residuals)?;
        validate_jacobian(&jacobian, m_residuals, n_params)?;
        let mut normal = NormalEquations::new(&jacobian, &residuals);

        // Scale the diagonal based on the Jacobian if needed
        // Use the column norms of the Jacobian to scale the parameters
        for j in 0..n_params {
            let col_norm = jacobian.col(j).norm_l2();

            if col_norm > T::zero() {
                diag[(j, 0)] = col_norm;
            }
        }

        // Initialize damping parameter (lambda)
        let tau = T::from_f64(self.tau).unwrap();

        let max_diag = diag.norm_max();

        let mut lambda = tau * if max_diag > T::zero() { max_diag } else { T::one() };

        // Main optimization loop
        let mut iterations = 0;
        let max_iterations = self.max_iterations;
        let epsilon_1 = T::from_f64(self.epsilon_1).unwrap();
        let epsilon_2 = T::from_f64(self.epsilon_2).unwrap();

        let mut residuals_norm = vector_norm(&residuals);

        // Initialize success flag for termination
        let mut success = false;
        let mut termination_reason = TerminationReason::MaxIterationsReached;
        let mut jacobian_is_current = true;
        let mut accepted_steps = 0;
        let mut rejected_steps = 0;

        while iterations < max_iterations {
            iterations += 1;

            let old_residuals_norm = residuals_norm;

            // Solve the trust region subproblem
            let update = calculate_parameter_update(&normal, lambda, &diag)?;

            validate_vector(&update.step, "Parameter update", Some(n_params))?;
            if !Float::is_finite(update.predicted_reduction) {
                return Err(Error::Numerical("Predicted reduction is not finite".to_string()));
            }

            // Check if the predicted reduction is too small
            if update.predicted_reduction <= T::zero() {
                if update.step_norm <= epsilon_2 * (vector_norm(&params) + epsilon_2) {
                    termination_reason = TerminationReason::SmallParameters;
                    success = true;
                    break;
                }
                rejected_steps += 1;
                lambda = adjust_lambda(lambda, T::zero(), false);
                continue;
            }

            // Apply the parameter update
            let mut new_params = params.clone();
            for i in 0..n_params {
                new_params[(i, 0)] += update.step[(i, 0)];
            }

            // Compute new residuals
            let new_residuals = problem.residuals(&new_params)?;
            validate_vector(&new_residuals, "Residuals", Some(m_residuals))?;
            let new_residuals_norm = vector_norm(&new_residuals);

            // Compute actual reduction
            let new_objective = calculate_objective_function(&new_residuals);
            if !Float::is_finite(new_objective) {
                return Err(Error::Numerical("Trial objective value is not finite".to_string()));
            }
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
            if !Float::is_finite(lambda) || lambda <= T::zero() {
                return Err(Error::Numerical("Damping parameter became invalid".to_string()));
            }

            // If this was a successful step, update the current solution
            if step_success {
                accepted_steps += 1;
                params = new_params;
                residuals = new_residuals;
                objective_function = new_objective;
                residuals_norm = new_residuals_norm;
                jacobian_is_current = false;

                if residuals_norm <= T::epsilon() {
                    termination_reason = TerminationReason::Converged;
                    success = true;
                    break;
                }

                let params_norm = vector_norm(&params);
                if let Some(reason) = check_convergence(old_residuals_norm, new_residuals_norm, update.step_norm, params_norm, epsilon_1, epsilon_2) {
                    termination_reason = reason;
                    success = true;
                    break;
                }

                // Recompute the Jacobian only after an accepted, nonterminal
                // step. Rejected damping attempts reuse the existing one.
                jacobian = calculator.calculate_jacobian_with_residuals(&problem, &params, &residuals)?;
                validate_jacobian(&jacobian, m_residuals, n_params)?;
                normal = NormalEquations::new(&jacobian, &residuals);
                jacobian_is_current = true;
            } else {
                rejected_steps += 1;
            }
        }

        if !jacobian_is_current {
            jacobian = calculator.calculate_jacobian_with_residuals(&problem, &params, &residuals)?;
            validate_jacobian(&jacobian, m_residuals, n_params)?;
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
            residual_evaluations: problem.residual_evaluations.get(),
            jacobian_evaluations: problem.jacobian_evaluations.get(),
            accepted_steps,
            rejected_steps,
            final_lambda: lambda,
        })
    }
}
