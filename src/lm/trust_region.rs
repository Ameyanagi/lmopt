use crate::{Error, Result};
use faer::{linalg::solvers::SolveLstsq, mat::Mat};
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::ops::{AddAssign, Mul};

/// Struct for the LM parameter update.
#[derive(Debug)]
pub(crate) struct LMParameterUpdate<T>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    pub step: Mat<T>,
    pub step_norm: T,
    pub predicted_reduction: T,
    pub used_svd_fallback: bool,
}

/// Quantities that remain unchanged while only the damping parameter changes.
pub(crate) struct Linearization<T>
where
    T: RealField + Copy,
{
    gradient: Mat<T>,
    column_norms: Mat<T>,
    augmented: Mat<T>,
    rhs: Mat<T>,
}

impl<T> Linearization<T>
where
    T: RealField + Copy + Mul<Output = T>,
{
    pub(crate) fn new(jacobian: &Mat<T>, residuals: &Mat<T>) -> Self {
        let m = jacobian.nrows();
        let n = jacobian.ncols();
        let mut augmented = Mat::zeros(m + n, n);
        for j in 0..n {
            for i in 0..m {
                augmented[(i, j)] = jacobian[(i, j)];
            }
        }

        let mut rhs = Mat::zeros(m + n, 1);
        for i in 0..m {
            rhs[(i, 0)] = -residuals[(i, 0)];
        }

        let column_norms = Mat::from_fn(n, 1, |j, _| jacobian.col(j).norm_l2());

        Self {
            gradient: jacobian.transpose().mul(residuals),
            column_norms,
            augmented,
            rhs,
        }
    }
}

impl<T> Linearization<T>
where
    T: RealField + Copy + Float,
{
    pub(crate) fn column_norm(&self, index: usize) -> T {
        self.column_norms[(index, 0)]
    }

    /// Infinity norm of the gradient after scaling it by the residual norm and
    /// the corresponding Jacobian-column norm. This is invariant when all
    /// residuals are multiplied by the same nonzero constant.
    pub(crate) fn scaled_gradient_norm(&self, residual_norm: T) -> T {
        if residual_norm == T::zero() {
            return T::zero();
        }

        let mut norm = T::zero();
        for i in 0..self.gradient.nrows() {
            let denominator = residual_norm * self.column_norms[(i, 0)];
            let gradient = self.gradient[(i, 0)];
            let component = if denominator > T::zero() && Float::is_finite(denominator) && Float::is_finite(gradient) {
                Float::abs(gradient) / denominator
            } else if self.column_norms[(i, 0)] > T::zero() {
                // Form the cosine from normalized vectors when Jᵀr or the
                // product of the norms over/underflows.
                let residual_rows = self.rhs.nrows() - self.gradient.nrows();
                let mut normalized_dot = T::zero();
                for row in 0..residual_rows {
                    normalized_dot = normalized_dot + (self.augmented[(row, i)] / self.column_norms[(i, 0)]) * (self.rhs[(row, 0)] / residual_norm);
                }
                Float::abs(normalized_dot)
            } else {
                T::zero()
            };
            if Float::is_finite(component) {
                norm = Float::max(norm, component);
            }
        }
        norm
    }
}

fn rank_threshold<T>(nrows: usize, ncols: usize, scale: T) -> Result<T>
where
    T: RealField + Copy + Float + FromPrimitive,
{
    let dimension = T::from_usize(nrows.max(ncols)).ok_or_else(|| Error::Numerical("matrix dimension cannot be represented in the selected scalar type".to_string()))?;
    Ok(scale * T::epsilon() * dimension)
}

fn solve_with_truncated_svd<T>(matrix: &Mat<T>, rhs: &Mat<T>) -> Result<Mat<T>>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    let svd = matrix.thin_svd().map_err(|error| Error::MatrixError(format!("augmented least-squares SVD failed: {error:?}")))?;
    let singular_values = svd.S();
    let size = matrix.nrows().min(matrix.ncols());
    let mut largest = T::zero();
    for k in 0..size {
        largest = Float::max(largest, Float::abs(singular_values[k]));
    }
    let threshold = rank_threshold(matrix.nrows(), matrix.ncols(), largest)?;

    let u = svd.U();
    let v = svd.V();
    let mut solution = Mat::zeros(matrix.ncols(), rhs.ncols());

    // Compute V * S^+ * U^T * rhs while truncating numerically zero
    // singular values. faer's general solve API intentionally performs an
    // exact reciprocal, so the tolerance must be applied here.
    for rhs_col in 0..rhs.ncols() {
        for k in 0..size {
            let singular_value = singular_values[k];
            if Float::abs(singular_value) <= threshold {
                continue;
            }

            let mut coefficient = T::zero();
            for i in 0..matrix.nrows() {
                coefficient += u[(i, k)] * rhs[(i, rhs_col)];
            }
            coefficient = coefficient / singular_value;

            for j in 0..matrix.ncols() {
                solution[(j, rhs_col)] += v[(j, k)] * coefficient;
            }
        }
    }

    Ok(solution)
}

fn solve_augmented_least_squares<T>(matrix: &Mat<T>, rhs: &Mat<T>) -> Result<(Mat<T>, bool)>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    let qr = matrix.col_piv_qr();
    let r = qr.thin_R();
    let n = matrix.ncols();
    let mut largest_diagonal = T::zero();
    for i in 0..n {
        largest_diagonal = Float::max(largest_diagonal, Float::abs(r[(i, i)]));
    }
    let threshold = rank_threshold(matrix.nrows(), matrix.ncols(), largest_diagonal)?;
    let full_rank = (0..n).all(|i| Float::abs(r[(i, i)]) > threshold);

    if full_rank {
        Ok((qr.solve_lstsq(rhs), false))
    } else {
        Ok((solve_with_truncated_svd(matrix, rhs)?, true))
    }
}

/// Calculate the parameter update by solving the damped least-squares system
/// directly:
///
/// ```text
/// min || J s + r ||² + lambda || D s ||²
/// ```
///
/// This avoids explicitly forming `J^T J`, which squares the condition number.
pub(crate) fn calculate_parameter_update<T>(linearization: &mut Linearization<T>, lambda: T, diag: &Mat<T>) -> Result<LMParameterUpdate<T>>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    if !Float::is_finite(lambda) || lambda < T::zero() {
        return Err(Error::Numerical("Damping parameter must be finite and non-negative".to_string()));
    }

    let m = linearization.rhs.nrows() - linearization.gradient.nrows();
    let n = linearization.gradient.nrows();

    let damping_scale = Float::sqrt(lambda);
    for j in 0..n {
        linearization.augmented[(m + j, j)] = damping_scale * diag[(j, 0)];
    }

    let (step, used_svd_fallback) = solve_augmented_least_squares(&linearization.augmented, &linearization.rhs)?;
    let mut step_norm = T::zero();
    for i in 0..n {
        step_norm = Float::hypot(step_norm, diag[(i, 0)] * step[(i, 0)]);
    }

    // From (J^T J + lambda D²)s = -J^T r, the quadratic-model reduction is
    // 0.5 * s^T(lambda D²s - J^T r).
    let half = T::one() / (T::one() + T::one());
    let mut predicted_reduction = T::zero();
    for i in 0..n {
        let damped_step = lambda * diag[(i, 0)] * diag[(i, 0)] * step[(i, 0)];
        predicted_reduction += half * step[(i, 0)] * (damped_step - linearization.gradient[(i, 0)]);
    }

    Ok(LMParameterUpdate {
        step,
        step_norm,
        predicted_reduction,
        used_svd_fallback,
    })
}

/// Adjust the damping parameter lambda based on the ratio of actual to predicted reduction.
pub(crate) fn adjust_lambda<T>(lambda: T, ratio: T, success: bool) -> T
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    let one = T::one();
    let two = one + one;
    let four = two + two;
    let half = one / two;
    let third = one / (two + one);
    let min_ratio = half * half;
    let max_ratio = one - min_ratio;

    if !success {
        return lambda * four;
    }

    if ratio < min_ratio {
        lambda * half.recip()
    } else if ratio > max_ratio {
        let mut new_lambda = lambda * third;
        let min_lambda = T::epsilon();
        if new_lambda < min_lambda {
            new_lambda = min_lambda;
        }
        new_lambda
    } else {
        lambda
    }
}

#[cfg(test)]
mod tests {
    use super::{adjust_lambda, calculate_parameter_update, Linearization};
    use anyhow::Result;
    use faer::Mat;

    #[test]
    fn solves_the_damped_least_squares_system() -> Result<()> {
        let residuals = Mat::<f64>::from_fn(2, 1, |i, _| if i == 0 { -1.0 } else { -2.0 });
        let jacobian = Mat::<f64>::identity(2, 2);
        let diagonal = Mat::<f64>::ones(2, 1);
        let linearization = Linearization::new(&jacobian, &residuals);

        let mut linearization = linearization;
        let undamped = calculate_parameter_update(&mut linearization, 0.0, &diagonal)?;
        assert!((undamped.step[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((undamped.step[(1, 0)] - 2.0).abs() < 1e-12);
        assert!((undamped.predicted_reduction - 2.5).abs() < 1e-12);
        assert!(!undamped.used_svd_fallback);

        let damped = calculate_parameter_update(&mut linearization, 1.0, &diagonal)?;
        assert!((damped.step[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((damped.step[(1, 0)] - 1.0).abs() < 1e-12);
        assert!((damped.predicted_reduction - 1.875).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn rank_deficient_system_uses_minimum_norm_svd_step() -> Result<()> {
        let jacobian = Mat::<f64>::from_fn(1, 2, |_, _| 1.0);
        let residuals = Mat::<f64>::from_fn(1, 1, |_, _| -2.0);
        let diagonal = Mat::<f64>::ones(2, 1);
        let linearization = Linearization::new(&jacobian, &residuals);

        let mut linearization = linearization;
        let update = calculate_parameter_update(&mut linearization, 0.0, &diagonal)?;

        assert!(update.used_svd_fallback);
        assert!((update.step[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((update.step[(1, 0)] - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn ill_conditioned_system_does_not_square_the_condition_number() -> Result<()> {
        let delta = 1e-10;
        let jacobian = Mat::<f64>::from_fn(3, 2, |i, j| match (i, j) {
            (_, 0) => 1.0,
            (0, 1) => 1.0,
            (1, 1) => 1.0 + delta,
            (2, 1) => 1.0 - delta,
            _ => unreachable!(),
        });
        let residuals = Mat::<f64>::from_fn(3, 1, |i, _| -(jacobian[(i, 0)] + 2.0 * jacobian[(i, 1)]));
        let diagonal = Mat::<f64>::ones(2, 1);
        let linearization = Linearization::new(&jacobian, &residuals);

        let mut linearization = linearization;
        let update = calculate_parameter_update(&mut linearization, 0.0, &diagonal)?;

        assert!((update.step[(0, 0)] - 1.0).abs() < 1e-5, "{:?}", update.step);
        assert!((update.step[(1, 0)] - 2.0).abs() < 1e-5, "{:?}", update.step);
        Ok(())
    }

    #[test]
    fn adjusts_lambda_from_step_quality() {
        assert!(adjust_lambda(1.0, 0.0, false) > 1.0);
        assert!(adjust_lambda(1.0, 0.1, true) > 1.0);
        assert!(adjust_lambda(1.0, 0.9, true) < 1.0);
        assert_eq!(adjust_lambda(1.0, 0.5, true), 1.0);
    }

    #[test]
    fn scaled_gradient_survives_intermediate_overflow_and_underflow() {
        for scale in [1e-200_f64, 1e200] {
            let jacobian = Mat::from_fn(1, 1, |_, _| scale);
            let residuals = Mat::from_fn(1, 1, |_, _| scale);
            let linearization = Linearization::new(&jacobian, &residuals);

            assert!((linearization.scaled_gradient_norm(residuals.norm_l2()) - 1.0).abs() < 1e-12);
        }
    }
}
