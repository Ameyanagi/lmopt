use crate::{Error, Result};
use faer::{linalg::solvers::Solve, mat::Mat, Side};
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::ops::{AddAssign, Mul};

/// Struct for the LM parameter update
#[derive(Debug)]
pub(crate) struct LMParameterUpdate<T>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    pub step: Mat<T>,
    pub step_norm: T,
    pub predicted_reduction: T,
}

/// Quantities that remain unchanged while only the damping parameter changes.
pub(crate) struct NormalEquations<T>
where
    T: RealField + Copy,
{
    jtj: Mat<T>,
    jtr: Mat<T>,
}

impl<T> NormalEquations<T>
where
    T: RealField + Copy + Mul<Output = T>,
{
    pub(crate) fn new(jacobian: &Mat<T>, residuals: &Mat<T>) -> Self {
        Self {
            jtj: jacobian.transpose().mul(jacobian),
            jtr: jacobian.transpose().mul(residuals),
        }
    }
}

/// Calculate the parameter update for a given lambda using the trust region approach
pub(crate) fn calculate_parameter_update<T>(normal: &NormalEquations<T>, lambda: T, diag: &Mat<T>) -> Result<LMParameterUpdate<T>>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    let n = normal.jtj.ncols();

    // Create the augmented matrix (J^T * J + lambda * diag^2)
    let mut augmented = normal.jtj.clone();
    for i in 0..n {
        augmented[(i, i)] += lambda * diag[(i, 0)] * diag[(i, 0)];
    }

    // Solve the symmetric positive-definite damped normal system with faer's
    // Cholesky implementation. Unlike the previous handwritten elimination,
    // this reports a factorization failure instead of dividing by a zero pivot.
    let mut b = normal.jtr.clone();
    for i in 0..b.nrows() {
        b[(i, 0)] = -b[(i, 0)];
    }
    let factor = augmented
        .llt(Side::Lower)
        .map_err(|error| Error::MatrixError(format!("damped normal matrix is not positive definite: {error:?}")))?;
    let step = factor.solve(&b);

    let step_norm = step.norm_l2();

    // From (JᵀJ + λD²)s = -Jᵀr, the quadratic-model reduction is
    // 0.5 * sᵀ(λD²s - Jᵀr).
    let half = T::from_f64(0.5).unwrap();
    let mut predicted_reduction = T::zero();
    for i in 0..n {
        let damped_step = lambda * diag[(i, 0)] * diag[(i, 0)] * step[(i, 0)];
        predicted_reduction += half * step[(i, 0)] * (damped_step - normal.jtr[(i, 0)]);
    }

    Ok(LMParameterUpdate { step, step_norm, predicted_reduction })
}

/// Adjust the damping parameter lambda based on the ratio of actual to predicted reduction
pub(crate) fn adjust_lambda<T>(lambda: T, ratio: T, success: bool) -> T
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    let four = T::from_f64(4.0).unwrap();
    let half = T::from_f64(0.5).unwrap();
    let third = T::from_f64(1.0 / 3.0).unwrap();
    let min_ratio = T::from_f64(0.25).unwrap();
    let max_ratio = T::from_f64(0.75).unwrap();

    if !success {
        // Increase lambda substantially
        return lambda * four;
    }

    if ratio < min_ratio {
        // Modest increase in lambda
        lambda * half.recip()
    } else if ratio > max_ratio {
        // Decrease lambda
        let mut new_lambda = lambda * third;

        // Ensure lambda doesn't become too small
        let min_lambda = T::from_f64(1e-10).unwrap();
        if new_lambda < min_lambda {
            new_lambda = min_lambda;
        }

        new_lambda
    } else {
        // Keep lambda the same
        lambda
    }
}

#[cfg(test)]
mod tests {
    use super::{adjust_lambda, calculate_parameter_update, NormalEquations};
    use faer::Mat;

    #[test]
    fn solves_the_damped_normal_system() {
        let residuals = Mat::<f64>::from_fn(2, 1, |i, _| if i == 0 { -1.0 } else { -2.0 });
        let jacobian = Mat::<f64>::identity(2, 2);
        let diagonal = Mat::<f64>::ones(2, 1);
        let normal = NormalEquations::new(&jacobian, &residuals);

        let undamped = calculate_parameter_update(&normal, 0.0, &diagonal).unwrap();
        assert!((undamped.step[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((undamped.step[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((undamped.predicted_reduction - 2.5).abs() < 1e-12);

        let damped = calculate_parameter_update(&normal, 1.0, &diagonal).unwrap();
        assert!((damped.step[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((damped.step[(1, 0)] - 1.0).abs() < 1e-12);
        assert!((damped.predicted_reduction - 1.875).abs() < 1e-12);
    }

    #[test]
    fn reports_a_singular_undamped_system() {
        let residuals = Mat::<f64>::ones(1, 1);
        let jacobian = Mat::<f64>::zeros(1, 1);
        let diagonal = Mat::<f64>::ones(1, 1);
        let normal = NormalEquations::new(&jacobian, &residuals);

        assert!(calculate_parameter_update(&normal, 0.0, &diagonal).is_err());
    }

    #[test]
    fn adjusts_lambda_from_step_quality() {
        assert!(adjust_lambda(1.0, 0.0, false) > 1.0);
        assert!(adjust_lambda(1.0, 0.1, true) > 1.0);
        assert!(adjust_lambda(1.0, 0.9, true) < 1.0);
        assert_eq!(adjust_lambda(1.0, 0.5, true), 1.0);
    }
}
