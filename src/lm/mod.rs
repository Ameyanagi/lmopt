mod algorithm;
mod convergence;
mod trust_region;

use crate::{LeastSquaresProblem, Result};
use faer::mat::Mat;
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
/// - `UserProvided`: Use the analytical Jacobian provided by the user through
///   the `jacobian()` method in the `LeastSquaresProblem` trait. This is
///   usually the fastest and most accurate approach.
///
/// - `AutoDiff`: Use automatic differentiation, which requires the nightly
///   compiler and the `autodiff` feature flag. This provides exact derivatives
///   without the need for manual calculation.
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
/// 1. If you can derive the analytical Jacobian, use `UserProvided`.
/// 2. If analytical Jacobian is difficult but you have nightly Rust,
///    consider `AutoDiff`.
/// 3. For a good balance of accuracy and performance with numerical
///    methods, use `NumericalCentral`.
/// 4. Use `NumericalForward` or `NumericalBackward` only when performance
///    is more important than accuracy, or when evaluating at boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianMethod {
    /// Use the user-provided Jacobian function.
    UserProvided,
    /// Use automatic differentiation (requires nightly and autodiff feature).
    AutoDiff,
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
/// - `SmallRelativeReduction`: The relative reduction in the residuals between
///   iterations became smaller than the specified tolerance (epsilon_1).
/// - `SmallParameters`: The relative change in parameters between iterations
///   became smaller than the specified tolerance (epsilon_2).
///
/// # Unsuccessful Termination
///
/// These reasons indicate that the algorithm stopped without finding
/// a satisfactory solution:
///
/// - `MaxIterationsReached`: The algorithm reached the maximum number of
///   iterations without meeting any convergence criteria.
/// - `InvalidJacobian`: The Jacobian matrix is invalid, ill-conditioned,
///   or could not be computed.
/// - `InvalidResiduals`: The residuals function returned invalid values
///   (e.g., NaN or Inf).
/// - `Other`: Any other reason for termination, with a descriptive message.
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
    /// The Jacobian matrix is invalid or ill-conditioned.
    InvalidJacobian,
    /// The residuals function returned invalid values.
    InvalidResiduals,
    /// Other reason for termination.
    Other(String),
}

impl TerminationReason {
    /// Check if the termination reason indicates success.
    pub fn is_success(&self) -> bool {
        matches!(self, TerminationReason::Converged | TerminationReason::SmallRelativeReduction | TerminationReason::SmallParameters)
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
/// ```rust,no_run
/// use lmopt::{LeastSquaresProblem, LevenbergMarquardt, Result};
///
/// struct MyProblem { /* ... */ }
///
/// impl LeastSquaresProblem<f64> for MyProblem {
///     // Implementation...
///     # fn residuals(&self, _: &faer::Mat<f64>) -> Result<faer::Mat<f64>> { todo!() }
/// }
///
/// fn main() -> Result<()> {
///     let problem = MyProblem { /* ... */ };
///     let initial_guess = faer::Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 });
///     let optimizer = LevenbergMarquardt::new();
///     
///     let result = optimizer.minimize(&problem, &initial_guess)?;
///     
///     // Check if optimization was successful
///     if result.success {
///         // Access the solution parameters
///         let optimal_params = result.solution_params;
///         println!("Found solution: {:?}", optimal_params);
///         
///         // Get the final value of the objective function
///         let objective_value = result.objective_function;
///         println!("Final objective value: {}", objective_value);
///         
///         // Performance metrics
///         println!("Iterations required: {}", result.iterations);
///         println!("Execution time: {:?}", result.execution_time);
///     } else {
///         // Analyze why optimization failed
///         println!("Optimization failed: {:?}", result.termination_reason);
///     }
///     
///     Ok(())
/// }
/// ```
#[derive(Debug)]
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
///     .with_epsilon_1(1e-6)  // Tolerance for relative reduction in residuals
///     .with_epsilon_2(1e-8)  // Tolerance for relative change in parameters
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

    /// Convergence tolerance for relative reduction in residuals.
    /// The algorithm terminates when the relative reduction in the residuals
    /// between iterations is less than this value.
    epsilon_1: f64,

    /// Convergence tolerance for relative change in parameters.
    /// The algorithm terminates when the relative change in parameters
    /// between iterations is less than this value.
    epsilon_2: f64,

    /// Initial value for the damping factor (λ).
    /// This controls the balance between gradient descent and Gauss-Newton approaches.
    /// A smaller value makes the algorithm behave more like Gauss-Newton initially.
    tau: f64,

    /// Method to use for calculating the Jacobian matrix.
    /// Controls whether to use analytical, automatic, or numerical differentiation.
    jacobian_method: JacobianMethod,

    /// Step size for numerical differentiation.
    /// Used when calculating the Jacobian numerically. A smaller value generally
    /// gives more accurate derivatives but may be affected by numerical precision.
    numerical_diff_step_size: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            epsilon_1: 1e-10,
            epsilon_2: 1e-10,
            tau: 1e-3,
            jacobian_method: JacobianMethod::NumericalCentral, // Changed from AutoDiff to NumericalCentral
            numerical_diff_step_size: 1e-6,
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

    /// Set the convergence tolerance for relative reduction in residuals.
    pub fn with_epsilon_1(mut self, epsilon_1: f64) -> Self {
        self.epsilon_1 = epsilon_1;
        self
    }

    /// Set the convergence tolerance for relative change in parameters.
    pub fn with_epsilon_2(mut self, epsilon_2: f64) -> Self {
        self.epsilon_2 = epsilon_2;
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

    /// Set the step size for numerical differentiation.
    pub fn with_numerical_diff_step_size(mut self, step_size: f64) -> Self {
        self.numerical_diff_step_size = step_size;
        self
    }

    /// Minimize the given least squares problem.
    pub fn minimize<T, P>(&self, problem: &P, initial_guess: &Mat<T>) -> Result<MinimizationReport<T>>
    where
        T: RealField + Copy + num_traits::Float + num_traits::FromPrimitive + AddAssign + 'static,
        P: LeastSquaresProblem<T>,
    {
        // Call the core implementation
        self.minimize_impl(problem, initial_guess)
    }
}
