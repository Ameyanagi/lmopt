//! Slice-and-closure convenience API.

use crate::{Error, LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, Result};
use faer::{Col, Mat};
use std::{fmt::Display, marker::PhantomData};

struct InfallibleClosureProblem<F> {
    residuals: F,
}

impl<F> LeastSquaresProblem<f64> for InfallibleClosureProblem<F>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let values = (self.residuals)(parameters.col_as_slice(0));
        Ok(Mat::from_fn(values.len(), 1, |i, _| values[i]))
    }
}

struct FallibleClosureProblem<F, E> {
    residuals: F,
    error: PhantomData<fn() -> E>,
}

impl<F, E> LeastSquaresProblem<f64> for FallibleClosureProblem<F, E>
where
    F: Fn(&[f64]) -> std::result::Result<Vec<f64>, E>,
    E: Display,
{
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let values = (self.residuals)(parameters.col_as_slice(0)).map_err(|error| Error::UserFunction(error.to_string()))?;
        Ok(Mat::from_fn(values.len(), 1, |i, _| values[i]))
    }
}

/// Minimize a least-squares problem defined by a closure.
///
/// This is the simplest entry point: parameters and residuals are ordinary
/// slices and vectors, and a numerical Jacobian is selected automatically.
/// Use [`LevenbergMarquardt::minimize_fn`] when custom solver settings are
/// needed.
///
/// # Example
///
/// ```
/// let xs = [0.0, 1.0, 2.0, 3.0];
/// let ys = [1.0, 3.0, 5.0, 7.0];
///
/// let fit = lmopt::least_squares(&[0.0, 0.0], |parameters| {
///     let [slope, intercept] = parameters else { return Vec::new() };
///     xs.iter()
///         .zip(ys)
///         .map(|(x, y)| slope * x + intercept - y)
///         .collect()
/// })?;
///
/// assert!((fit.parameters()[0] - 2.0).abs() < 1e-6);
/// assert!((fit.parameters()[1] - 1.0).abs() < 1e-6);
/// # Ok::<(), lmopt::Error>(())
/// ```
pub fn least_squares<F>(initial_guess: &[f64], residuals: F) -> Result<MinimizationReport<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    LevenbergMarquardt::new().minimize_fn(initial_guess, residuals)
}

/// Fallible variant of [`least_squares`].
///
/// The callback's error can be any type implementing [`Display`]; its message
/// is returned as [`crate::Error::UserFunction`].
pub fn try_least_squares<F, E>(initial_guess: &[f64], residuals: F) -> Result<MinimizationReport<f64>>
where
    F: Fn(&[f64]) -> std::result::Result<Vec<f64>, E>,
    E: Display,
{
    LevenbergMarquardt::new().try_minimize_fn(initial_guess, residuals)
}

impl LevenbergMarquardt {
    /// Minimize residuals produced by an infallible slice-based closure.
    pub fn minimize_fn<F>(&self, initial_guess: &[f64], residuals: F) -> Result<MinimizationReport<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let problem = InfallibleClosureProblem { residuals };
        let initial_guess = Col::from_fn(initial_guess.len(), |i| initial_guess[i]);
        self.minimize(&problem, &initial_guess)
    }

    /// Run slice-based residuals and return a report even on non-convergence.
    pub fn optimize_fn<F>(&self, initial_guess: &[f64], residuals: F) -> Result<MinimizationReport<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let problem = InfallibleClosureProblem { residuals };
        let initial_guess = Col::from_fn(initial_guess.len(), |i| initial_guess[i]);
        self.optimize(&problem, &initial_guess)
    }

    /// Minimize residuals produced by a fallible slice-based closure.
    pub fn try_minimize_fn<F, E>(&self, initial_guess: &[f64], residuals: F) -> Result<MinimizationReport<f64>>
    where
        F: Fn(&[f64]) -> std::result::Result<Vec<f64>, E>,
        E: Display,
    {
        let problem = FallibleClosureProblem { residuals, error: PhantomData };
        let initial_guess = Col::from_fn(initial_guess.len(), |i| initial_guess[i]);
        self.minimize(&problem, &initial_guess)
    }

    /// Run fallible slice-based residuals and return a report even on
    /// non-convergence.
    pub fn try_optimize_fn<F, E>(&self, initial_guess: &[f64], residuals: F) -> Result<MinimizationReport<f64>>
    where
        F: Fn(&[f64]) -> std::result::Result<Vec<f64>, E>,
        E: Display,
    {
        let problem = FallibleClosureProblem { residuals, error: PhantomData };
        let initial_guess = Col::from_fn(initial_guess.len(), |i| initial_guess[i]);
        self.optimize(&problem, &initial_guess)
    }
}
