mod algorithm;
mod convergence;
mod trust_region;

use crate::{LeastSquaresProblem, Result};
use faer::mat::{AsMatRef, Mat};
use faer_traits::RealField;
use std::ops::AddAssign;

/// Methods for calculating the Jacobian matrix.
///
/// The Jacobian matrix contains the partial derivatives of each residual
/// function with respect to each parameter. There are several methods to
/// compute this matrix:
///
/// # Methods
///
/// - `Auto`: Use an analytical Jacobian when the problem supplies one,
///   otherwise use central finite differences. This is the default.
///
/// - `UserProvided`: Use the analytical Jacobian provided by the user through
///   the `jacobian()` method in the `LeastSquaresProblem` trait. This is
///   usually the fastest and most accurate approach.
///
/// - `NumericalCentral`: Use central difference approximation:
///   `(f(x+h) - f(x-h)) / (2*h)`. This is more accurate than forward or
///   backward differences but requires twice as many function evaluations.
///
/// - `NumericalForward`: Use forward difference approximation:
///   `(f(x+h) - f(x)) / h`. This is faster than central differences but
///   less accurate.
///
/// - `NumericalBackward`: Use backward difference approximation:
///   `(f(x) - f(x-h)) / h`. Similar to forward differences but evaluates
///   at x-h instead of x+h.
///
/// # Choosing a Method
///
/// 1. Prefer `Auto` unless you need to force a specific strategy.
/// 2. If you can derive the analytical Jacobian, use `UserProvided`.
/// 3. For a good balance of accuracy and performance with numerical
///    methods, use `NumericalCentral`.
/// 4. Use `NumericalForward` or `NumericalBackward` only when performance
///    is more important than accuracy, or when evaluating at boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianMethod {
    /// Prefer a user-provided Jacobian and otherwise use central differences.
    Auto,
    /// Use the user-provided Jacobian function.
    UserProvided,
    /// Use numerical differentiation with central differences.
    NumericalCentral,
    /// Use numerical differentiation with forward differences.
    NumericalForward,
    /// Use numerical differentiation with backward differences.
    NumericalBackward,
}

/// Reasons for terminating the minimization process.
///
/// This enum provides detailed information about why the optimization
/// algorithm stopped. It's useful for diagnosing issues with convergence
/// or understanding the quality of the solution.
///
/// # Successful Termination
///
/// These reasons indicate that the algorithm found a solution that
/// meets one of the convergence criteria:
///
/// - `Converged`: The algorithm successfully converged to a solution.
/// - `SmallRelativeReduction`: The relative reduction in the objective between
///   iterations became smaller than `ftol`.
/// - `SmallParameters`: The relative change in parameters between iterations
///   became smaller than `xtol`.
/// - `SmallGradient`: The scaled gradient infinity norm became smaller than
///   `gtol`.
///
/// # Unsuccessful Termination
///
/// These reasons indicate that the algorithm stopped without finding
/// a satisfactory solution:
///
/// - `MaxIterationsReached`: The algorithm reached the maximum number of
///   iterations without meeting any convergence criteria.
/// - `NoProgress`: Floating-point precision prevented a proposed step from
///   changing the parameters or objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// The algorithm converged successfully.
    Converged,
    /// Maximum number of iterations reached.
    MaxIterationsReached,
    /// Reached a solution with a small relative reduction in the residuals.
    SmallRelativeReduction,
    /// Reached a solution with small parameter changes.
    SmallParameters,
    /// Reached a point with a small scaled gradient.
    SmallGradient,
    /// The solver could not make representable progress.
    NoProgress,
}

impl TerminationReason {
    /// Check if the termination reason indicates success.
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            TerminationReason::Converged | TerminationReason::SmallRelativeReduction | TerminationReason::SmallParameters | TerminationReason::SmallGradient
        )
    }
}

/// Results and diagnostics from a minimization run.
///
/// This struct contains the solution to the optimization problem along
/// with detailed information about the minimization process. It allows
/// you to analyze the quality of the solution and understand how the
/// algorithm performed.
///
/// # Example
///
/// ```rust
/// let result = lmopt::least_squares(&[0.0], |parameters| {
///     vec![parameters[0] - 2.0]
/// })?;
///
/// // `least_squares` and `minimize` only return successful reports.
/// println!("Found solution: {:?}", result.parameters());
/// println!("Final objective value: {}", result.objective_function);
/// println!("Iterations required: {}", result.iterations);
/// # Ok::<(), lmopt::Error>(())
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct MinimizationReport<T>
where
    T: RealField + Copy,
{
    /// The optimized parameters (solution vector).
    pub solution_params: Mat<T>,

    /// The residuals at the solution.
    /// These are the differences between the model predictions and the observed data.
    pub residuals: Mat<T>,

    /// The objective function value at the solution (0.5 * ||residuals||²).
    /// Lower values indicate a better fit.
    pub objective_function: T,

    /// The number of iterations the algorithm performed.
    /// A very low number might indicate quick convergence or an issue.
    /// A number equal to max_iterations suggests the algorithm may not have converged.
    pub iterations: usize,

    /// The Jacobian matrix at the solution, if available.
    /// This contains the partial derivatives of each residual with respect to each parameter.
    pub jacobian: Option<Mat<T>>,

    /// The method that was used to calculate the Jacobian.
    pub jacobian_method_used: JacobianMethod,

    /// Whether the minimization was successful (`true`) or not (`false`).
    /// This is based on the termination reason.
    pub success: bool,

    /// The specific reason why the algorithm terminated.
    /// Use this to diagnose convergence issues or understand the quality of the solution.
    pub termination_reason: TerminationReason,

    /// The total time taken for the minimization process.
    /// Useful for benchmarking and optimization.
    pub execution_time: std::time::Duration,

    /// Number of calls to the user residual function.
    pub residual_evaluations: usize,

    /// Number of calls to the user analytical Jacobian function.
    pub jacobian_evaluations: usize,

    /// Number of trial steps accepted by the trust-region logic.
    pub accepted_steps: usize,

    /// Number of trial steps rejected by the trust-region logic.
    pub rejected_steps: usize,

    /// Final damping parameter.
    pub final_lambda: T,

    /// Number of rank-deficient linearized systems solved with truncated SVD.
    pub svd_fallbacks: usize,
}

impl<T> MinimizationReport<T>
where
    T: RealField + Copy,
{
    /// Borrow the optimized parameters as a plain slice.
    ///
    /// This is the easiest way to consume a result without using faer indexing.
    pub fn parameters(&self) -> &[T] {
        self.solution_params.col_as_slice(0)
    }

    /// Whether the termination reason indicates convergence.
    pub fn converged(&self) -> bool {
        self.termination_reason.is_success()
    }
}

/// Configuration for the Levenberg-Marquardt optimization algorithm.
///
/// This struct provides a fluent API for configuring the behavior of the
/// Levenberg-Marquardt algorithm. It allows you to customize convergence
/// criteria, numerical methods, and other aspects of the optimization process.
///
/// # Example
///
/// ```rust,no_run
/// use lmopt::{LevenbergMarquardt, JacobianMethod};
///
/// // Create an optimizer with default settings
/// let default_optimizer = LevenbergMarquardt::new();
///
/// // Create a customized optimizer
/// let custom_optimizer = LevenbergMarquardt::new()
///     .with_max_iterations(200)
///     .with_ftol(1e-6)       // Relative objective-reduction tolerance
///     .with_xtol(1e-8)       // Relative scaled-parameter-step tolerance
///     .with_gtol(1e-8)       // Scaled-gradient tolerance
///     .with_tau(1e-4)        // Initial damping parameter
///     .with_jacobian_method(JacobianMethod::NumericalCentral)
///     .with_numerical_diff_step_size(1e-5);
/// ```
///
/// # Algorithm Details
///
/// The Levenberg-Marquardt algorithm combines gradient descent and Gauss-Newton
/// methods, making it robust and efficient for nonlinear least squares problems.
/// It uses a damping parameter (λ) that is adjusted during optimization:
///
/// - When λ is large, the algorithm behaves like gradient descent (more stable but slower)
/// - When λ is small, it behaves like Gauss-Newton (faster but may diverge)
///
/// The algorithm adaptively adjusts λ based on whether the current step reduces
/// the objective function.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardt {
    /// Maximum number of iterations.
    /// The algorithm will stop after this many iterations even if not converged.
    max_iterations: usize,

    /// Convergence tolerance for relative reduction in the objective.
    /// The algorithm terminates when the relative reduction in the objective
    /// between iterations is less than this value.
    ftol: Option<f64>,

    /// Convergence tolerance for relative change in parameters.
    /// The algorithm terminates when the relative change in parameters
    /// between iterations is less than this value.
    xtol: Option<f64>,

    /// Convergence tolerance for the scaled gradient infinity norm.
    gtol: Option<f64>,

    /// Initial value for the damping factor (λ).
    /// This controls the balance between gradient descent and Gauss-Newton approaches.
    /// A smaller value makes the algorithm behave more like Gauss-Newton initially.
    tau: f64,

    /// Method to use for calculating the Jacobian matrix.
    /// Controls whether to use analytical, automatic, or numerical differentiation.
    jacobian_method: JacobianMethod,

    /// Optional relative step size for numerical differentiation.
    /// When omitted, the solver selects a precision-aware default for the scalar
    /// type and finite-difference method.
    numerical_diff_step_size: Option<f64>,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            ftol: None,
            xtol: None,
            gtol: None,
            tau: 1e-3,
            jacobian_method: JacobianMethod::Auto,
            numerical_diff_step_size: None,
        }
    }
}

impl LevenbergMarquardt {
    /// Create a new Levenberg-Marquardt algorithm with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Backward-compatible alias for [`Self::with_ftol`].
    #[deprecated(since = "0.3.0", note = "use with_ftol")]
    pub fn with_epsilon_1(self, epsilon_1: f64) -> Self {
        self.with_ftol(epsilon_1)
    }

    /// Set the relative objective-reduction tolerance.
    pub fn with_ftol(mut self, ftol: f64) -> Self {
        self.ftol = Some(ftol);
        self
    }

    /// Backward-compatible alias for [`Self::with_xtol`].
    #[deprecated(since = "0.3.0", note = "use with_xtol")]
    pub fn with_epsilon_2(self, epsilon_2: f64) -> Self {
        self.with_xtol(epsilon_2)
    }

    /// Set the relative parameter-step tolerance.
    pub fn with_xtol(mut self, xtol: f64) -> Self {
        self.xtol = Some(xtol);
        self
    }

    /// Set the scaled-gradient convergence tolerance.
    pub fn with_gtol(mut self, gtol: f64) -> Self {
        self.gtol = Some(gtol);
        self
    }

    /// Set the initial value for the damping factor.
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Set the method to use for calculating the Jacobian.
    pub fn with_jacobian_method(mut self, method: JacobianMethod) -> Self {
        self.jacobian_method = method;
        self
    }

    /// Set the relative step size for numerical differentiation.
    pub fn with_numerical_diff_step_size(mut self, step_size: f64) -> Self {
        self.numerical_diff_step_size = Some(step_size);
        self
    }

    /// Run the optimizer and return its report, including a partial result when
    /// the convergence criteria are not met.
    ///
    /// Evaluation, configuration, and numerical failures are returned as
    /// errors. Inspect [`MinimizationReport::converged`] for the termination
    /// status. Use [`Self::minimize`] when non-convergence should also be an
    /// error.
    pub fn optimize<T, P, I>(&self, problem: &P, initial_guess: &I) -> Result<MinimizationReport<T>>
    where
        T: RealField + Copy + num_traits::Float + num_traits::FromPrimitive + AddAssign + 'static,
        P: LeastSquaresProblem<T>,
        I: AsMatRef<T = T> + ?Sized,
    {
        let initial_guess = initial_guess.as_mat_ref().as_dyn().to_owned();
        self.minimize_impl(problem, initial_guess)
    }

    /// Minimize the given least-squares problem.
    ///
    /// Unlike [`Self::optimize`], this checked convenience method returns
    /// [`crate::Error::NoConvergence`] when the iteration limit is reached or
    /// floating-point precision prevents further progress.
    pub fn minimize<T, P, I>(&self, problem: &P, initial_guess: &I) -> Result<MinimizationReport<T>>
    where
        T: RealField + Copy + num_traits::Float + num_traits::FromPrimitive + AddAssign + 'static,
        P: LeastSquaresProblem<T>,
        I: AsMatRef<T = T> + ?Sized,
    {
        let report = self.optimize(problem, initial_guess)?;
        if report.converged() {
            Ok(report)
        } else {
            Err(crate::Error::NoConvergence {
                reason: report.termination_reason,
                iterations: report.iterations,
            })
        }
    }
}
