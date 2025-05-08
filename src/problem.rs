use crate::Result;
use faer::mat::Mat;
use faer_traits::RealField;

/// A trait for defining a nonlinear least squares problem.
///
/// This trait represents a mathematical problem of the form:
///
/// ```text
/// minimize f(x) = 0.5 * sum(r_i(x)^2)
/// ```
///
/// Where:
/// - `x` is a vector of parameters
/// - `r_i(x)` are the residual functions
/// - `f(x)` is the objective function (sum of squared residuals)
///
/// To use the Levenberg-Marquardt algorithm, you must implement this trait
/// for your specific problem, providing at minimum the `residuals` function.
/// Optionally, you can provide an analytical `jacobian` function for better
/// performance and accuracy.
///
/// # Example
///
/// ```rust
/// use lmopt::{LeastSquaresProblem, Result};
/// use faer::Mat;
///
/// // A simple linear model: y = a*x + b
/// struct LinearModel {
///     // Data points
///     x_data: Vec<f64>,
///     y_data: Vec<f64>,
/// }
///
/// impl LeastSquaresProblem<f64> for LinearModel {
///     fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
///         let a = parameters[(0, 0)];
///         let b = parameters[(1, 0)];
///         
///         let n = self.x_data.len();
///         let mut residuals = Mat::zeros(n, 1);
///         
///         // Calculate residuals: r_i = (a*x_i + b) - y_i
///         for i in 0..n {
///             let x = self.x_data[i];
///             let y = self.y_data[i];
///             residuals[(i, 0)] = a * x + b - y;
///         }
///         
///         Ok(residuals)
///     }
///     
///     fn jacobian(&self, _parameters: &Mat<f64>) -> Option<Mat<f64>> {
///         let n = self.x_data.len();
///         let mut jacobian = Mat::zeros(n, 2);
///         
///         // Calculate derivatives:
///         // ∂r_i/∂a = x_i
///         // ∂r_i/∂b = 1
///         for i in 0..n {
///             jacobian[(i, 0)] = self.x_data[i];
///             jacobian[(i, 1)] = 1.0;
///         }
///         
///         Some(jacobian)
///     }
/// }
/// ```
pub trait LeastSquaresProblem<T>
where
    T: RealField + Copy,
{
    /// Compute the residuals for the given parameters.
    ///
    /// The residuals represent the difference between the model predictions
    /// and the observed data. For a good fit, residuals should be close to zero.
    ///
    /// # Parameters
    ///
    /// * `parameters` - Column vector of parameters (shape n×1)
    ///
    /// # Returns
    ///
    /// * A column vector of residuals (shape m×1)
    ///
    /// # Notes
    ///
    /// - The number of rows in the residuals matrix should be equal to the number of data points
    /// - The number of columns should be 1 (it must be a column vector)
    /// - The size of the residuals vector should be consistent for all parameter values
    fn residuals(&self, parameters: &Mat<T>) -> Result<Mat<T>>;

    /// Optionally compute the Jacobian matrix for the given parameters.
    ///
    /// The Jacobian matrix contains the partial derivatives of each residual
    /// with respect to each parameter:
    ///
    /// ```text
    /// J[i,j] = ∂r_i/∂p_j
    /// ```
    ///
    /// Where:
    /// - `r_i` is the i-th residual
    /// - `p_j` is the j-th parameter
    ///
    /// # Parameters
    ///
    /// * `parameters` - Column vector of parameters (shape n×1)
    ///
    /// # Returns
    ///
    /// * `Some(Mat<T>)` - Jacobian matrix of shape m×n, where m is the number
    ///   of residuals and n is the number of parameters
    /// * `None` - If no analytical Jacobian is provided (the algorithm will use
    ///   automatic or numerical differentiation instead)
    ///
    /// # Notes
    ///
    /// - For performance reasons, providing an analytical Jacobian is highly
    ///   recommended for complex problems
    /// - The Jacobian must have the correct dimensions: m rows (matching residuals)
    ///   and n columns (matching parameters)
    fn jacobian(&self, _parameters: &Mat<T>) -> Option<Mat<T>> {
        None
    }

    /// Hint whether to use autodiff or numerical differentiation when no Jacobian is provided.
    ///
    /// This hint is used when no analytical Jacobian is provided via the `jacobian()`
    /// method, to choose between automatic differentiation and numerical differentiation.
    ///
    /// # Returns
    ///
    /// * `true` - Prefer automatic differentiation (when available)
    /// * `false` - Prefer numerical differentiation
    ///
    /// # Notes
    ///
    /// - The automatic differentiation feature requires the nightly compiler
    ///   and the 'autodiff' feature flag
    /// - If autodiff is not available, the system will fall back to numerical
    ///   differentiation regardless of this hint
    /// - Default is `true` (prefer autodiff when available)
    fn prefer_autodiff(&self) -> bool {
        true
    }
}
