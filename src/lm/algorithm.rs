use super::{
    convergence::check_convergence,
    trust_region::{adjust_lambda, calculate_parameter_update, Linearization},
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

// Calculate ||residuals||^2 / 2 from an already-computed norm.
fn objective_from_residual_norm<T>(norm: T) -> T
where
    T: RealField + Copy,
{
    let two = T::one() + T::one();
    norm * norm / two
}

// Calculate the norm of a vector
fn vector_norm<T>(vec: &Mat<T>) -> T
where
    T: RealField + Copy + Float,
{
    vec.norm_l2()
}

fn scaled_vector_norm<T>(vector: &Mat<T>, diagonal: &Mat<T>) -> T
where
    T: RealField + Copy + Float,
{
    let mut norm = T::zero();
    for i in 0..vector.nrows() {
        norm = Float::hypot(norm, diagonal[(i, 0)] * vector[(i, 0)]);
    }
    norm
}

fn default_numerical_step<T>(method: crate::JacobianMethod) -> f64
where
    T: RealField + Copy + Float + FromPrimitive,
{
    // Central differences have O(h²) truncation error, while one-sided
    // differences have O(h). Choose precision-aware defaults for the scalar
    // types supported in practice without forcing callers to tune them.
    let is_single_precision = std::mem::size_of::<T>() <= std::mem::size_of::<f32>();
    match method {
        crate::JacobianMethod::Auto | crate::JacobianMethod::NumericalCentral => {
            if is_single_precision {
                5e-3
            } else {
                6e-6
            }
        }
        crate::JacobianMethod::NumericalForward | crate::JacobianMethod::NumericalBackward => {
            if is_single_precision {
                3e-4
            } else {
                1.5e-8
            }
        }
        crate::JacobianMethod::UserProvided => 1.0,
    }
}

fn default_tolerance<T>() -> f64
where
    T: RealField + Copy + Float + FromPrimitive,
{
    if std::mem::size_of::<T>() <= std::mem::size_of::<f32>() {
        1e-5
    } else {
        1e-10
    }
}

fn scalar_from_f64<T>(value: f64, name: &str) -> Result<T>
where
    T: RealField + Copy + FromPrimitive,
{
    T::from_f64(value).ok_or_else(|| Error::Numerical(format!("failed to represent {name} in the selected scalar type")))
}

impl LevenbergMarquardt {
    /// Core implementation of the Levenberg-Marquardt algorithm
    pub(crate) fn minimize_impl<T, P>(&self, problem: &P, initial_guess: Mat<T>) -> Result<MinimizationReport<T>>
    where
        T: RealField + Copy + Float + FromPrimitive + AddAssign + 'static,
        P: LeastSquaresProblem<T>,
    {
        let start_time = Instant::now();

        if self.max_iterations == 0 {
            return Err(Error::InvalidParameter("max_iterations must be greater than zero".to_string()));
        }
        if let Some(ftol) = self.ftol {
            validate_positive_finite(ftol, "ftol")?;
        }
        if let Some(xtol) = self.xtol {
            validate_positive_finite(xtol, "xtol")?;
        }
        if let Some(gtol) = self.gtol {
            validate_positive_finite(gtol, "gtol")?;
        }
        validate_positive_finite(self.tau, "tau")?;
        if let Some(step_size) = self.numerical_diff_step_size {
            if !matches!(self.jacobian_method, crate::JacobianMethod::UserProvided) {
                validate_positive_finite(step_size, "numerical_diff_step_size")?;
            }
        }

        validate_vector(&initial_guess, "Initial guess", None)?;
        let problem = CountingProblem::new(problem);

        // Set up variables
        let mut params = initial_guess;
        let n_params = params.nrows();

        // Compute initial residuals
        let mut residuals = problem.residuals(&params)?;
        let m_residuals = residuals.nrows();
        validate_vector(&residuals, "Residuals", None)?;

        // Calculate the residual norm once; both convergence and the objective
        // use it throughout the iteration.
        let mut residuals_norm = vector_norm(&residuals);
        let mut objective_function = objective_from_residual_norm(residuals_norm);
        if !Float::is_finite(objective_function) {
            return Err(Error::Numerical("Initial objective value is not finite".to_string()));
        }

        // Avoid trying to factor a singular system when the initial point is
        // already an exact solution.
        if residuals_norm == T::zero() {
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
                svd_fallbacks: 0,
            });
        }

        // Initialize the scaling diagonal matrix
        let mut diag = Mat::ones(n_params, 1);

        // Calculate initial jacobian
        let numerical_diff_step_size = match self.numerical_diff_step_size {
            Some(step_size) => step_size,
            None => default_numerical_step::<T>(self.jacobian_method),
        };
        let calculator = get_jacobian_calculator(self.jacobian_method, numerical_diff_step_size);
        let mut jacobian = calculator.calculate_jacobian_with_residuals(&problem, &params, &residuals)?;
        validate_jacobian(&jacobian, m_residuals, n_params)?;
        let mut linearization = Linearization::new(&jacobian, &residuals);

        // Scale the diagonal based on the Jacobian if needed
        // Use the column norms of the Jacobian to scale the parameters
        for j in 0..n_params {
            let col_norm = linearization.column_norm(j);

            if col_norm > T::zero() {
                diag[(j, 0)] = col_norm;
            }
        }

        // D contains Jacobian-column norms, so lambda is dimensionless in
        // lambda * D². Multiplying lambda by another Jacobian norm here would
        // make the algorithm depend on the arbitrary scale of the residuals.
        let tau = scalar_from_f64(self.tau, "tau")?;
        let mut lambda = tau;

        // Main optimization loop
        let mut iterations = 0;
        let max_iterations = self.max_iterations;
        let default_tolerance = default_tolerance::<T>();
        let ftol = scalar_from_f64(
            match self.ftol {
                Some(value) => value,
                None => default_tolerance,
            },
            "ftol",
        )?;
        let xtol = scalar_from_f64(
            match self.xtol {
                Some(value) => value,
                None => default_tolerance,
            },
            "xtol",
        )?;
        let gtol = scalar_from_f64(
            match self.gtol {
                Some(value) => value,
                None => default_tolerance,
            },
            "gtol",
        )?;

        // Track termination and diagnostics.
        let mut termination_reason = TerminationReason::MaxIterationsReached;
        let mut jacobian_is_current = true;
        let mut accepted_steps = 0;
        let mut rejected_steps = 0;
        let mut svd_fallbacks = 0;

        if linearization.scaled_gradient_norm(residuals_norm) <= gtol {
            return Ok(MinimizationReport {
                solution_params: params,
                residuals,
                objective_function,
                iterations: 0,
                jacobian: Some(jacobian),
                jacobian_method_used: calculator.method_used(),
                success: true,
                termination_reason: TerminationReason::SmallGradient,
                execution_time: start_time.elapsed(),
                residual_evaluations: problem.residual_evaluations.get(),
                jacobian_evaluations: problem.jacobian_evaluations.get(),
                accepted_steps,
                rejected_steps,
                final_lambda: lambda,
                svd_fallbacks,
            });
        }

        while iterations < max_iterations {
            iterations += 1;

            let old_objective = objective_function;

            // Solve the trust region subproblem
            let update = calculate_parameter_update(&mut linearization, lambda, &diag)?;
            svd_fallbacks += usize::from(update.used_svd_fallback);

            validate_vector(&update.step, "Parameter update", Some(n_params))?;
            if !Float::is_finite(update.predicted_reduction) {
                return Err(Error::Numerical("Predicted reduction is not finite".to_string()));
            }

            // Check if the predicted reduction is too small
            if update.predicted_reduction <= T::zero() {
                rejected_steps += 1;
                lambda = adjust_lambda(lambda, T::zero(), false);
                if !Float::is_finite(lambda) || lambda <= T::zero() {
                    return Err(Error::Numerical("Damping parameter became invalid".to_string()));
                }
                continue;
            }

            // Apply the parameter update
            let mut new_params = params.clone();
            for i in 0..n_params {
                new_params[(i, 0)] += update.step[(i, 0)];
            }

            if (0..n_params).all(|i| new_params[(i, 0)] == params[(i, 0)]) {
                rejected_steps += 1;
                termination_reason = TerminationReason::NoProgress;
                break;
            }

            // Compute new residuals
            let new_residuals = problem.residuals(&new_params)?;
            validate_vector(&new_residuals, "Residuals", Some(m_residuals))?;
            let new_residuals_norm = vector_norm(&new_residuals);

            // Compute actual reduction
            let new_objective = objective_from_residual_norm(new_residuals_norm);
            if !Float::is_finite(new_objective) {
                return Err(Error::Numerical("Trial objective value is not finite".to_string()));
            }
            let actual_reduction = objective_function - new_objective;

            // Calculate ratio of actual to predicted reduction
            let ratio = actual_reduction / update.predicted_reduction;

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

                if residuals_norm == T::zero() {
                    termination_reason = TerminationReason::Converged;
                    break;
                }

                let params_norm = scaled_vector_norm(&params, &diag);
                if let Some(reason) = check_convergence(old_objective, new_objective, update.step_norm, params_norm, ftol, xtol) {
                    termination_reason = reason;
                    break;
                }

                // Recompute the Jacobian only after an accepted, nonterminal
                // step. Rejected damping attempts reuse the existing one.
                jacobian = calculator.calculate_jacobian_with_residuals(&problem, &params, &residuals)?;
                validate_jacobian(&jacobian, m_residuals, n_params)?;
                linearization = Linearization::new(&jacobian, &residuals);
                for j in 0..n_params {
                    diag[(j, 0)] = Float::max(diag[(j, 0)], linearization.column_norm(j));
                }
                jacobian_is_current = true;

                if linearization.scaled_gradient_norm(residuals_norm) <= gtol {
                    termination_reason = TerminationReason::SmallGradient;
                    break;
                }
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
            success: termination_reason.is_success(),
            termination_reason,
            execution_time,
            residual_evaluations: problem.residual_evaluations.get(),
            jacobian_evaluations: problem.jacobian_evaluations.get(),
            accepted_steps,
            rejected_steps,
            final_lambda: lambda,
            svd_fallbacks,
        })
    }
}
