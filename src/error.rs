use thiserror::Error;

/// Error type for lmopt operations
#[derive(Error, Debug)]
pub enum Error {
    /// Error in user-provided function (residuals or jacobian)
    #[error("User function failed: {0}")]
    UserFunction(String),

    /// Numerical error (e.g., NaN, Inf)
    #[error("Numerical error: {0}")]
    Numerical(String),

    /// Dimensional mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error in matrix operations
    #[error("Matrix error: {0}")]
    MatrixError(String),

    /// No convergence achieved
    #[error("No convergence: {0}")]
    NoConvergence(String),

    /// Automatic differentiation was requested but no backend is available.
    #[error("Automatic differentiation is unavailable: {0}")]
    AutoDiffUnavailable(String),
}

/// Result type for lmopt operations
pub type Result<T> = std::result::Result<T, Error>;
