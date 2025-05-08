use crate::Result;
use faer::mat::Mat;
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::ops::{AddAssign, Mul};

/// Struct for the LM parameter update
#[derive(Debug)]
pub(crate) struct LMParameterUpdate<T>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    pub lambda: T,
    pub step: Mat<T>,
    pub step_norm: T,
    pub predicted_reduction: T,
}

/// Calculate the parameter update for a given lambda using the trust region approach
pub(crate) fn calculate_parameter_update<T>(jacobian: &Mat<T>, residuals: &Mat<T>, lambda: T, diag: &Mat<T>) -> Result<LMParameterUpdate<T>>
where
    T: RealField + Copy + Float + FromPrimitive + AddAssign,
{
    let n = jacobian.ncols();

    // Compute J^T * J and J^T * r
    let jtj = jacobian.transpose().mul(jacobian);
    let jtr = jacobian.transpose().mul(residuals);

    // Create the augmented matrix (J^T * J + lambda * diag^2)
    let mut augmented = jtj.clone();
    for i in 0..n {
        augmented[(i, i)] += lambda * diag[(i, 0)] * diag[(i, 0)];
    }

    // Solve the linear system (J^T * J + lambda * diag^2) * step = J^T * r
    let mut step = Mat::zeros(n, 1);

    // Use direct matrix inversion as a fallback approach
    // Implement a simple solution using matrix inversion
    // In a real implementation, we would use more efficient solvers

    // Create an identity matrix for inversion if needed
    let _identity = Mat::<T>::identity(n, n);

    // Manually solve the linear system (J^T * J + lambda * diag^2) * x = -J^T * r
    // using a simplistic approach

    // First, apply Gaussian elimination to solve the system
    // For simplicity, just copy the right-hand side
    let mut b = jtr.clone();

    // Make all entries of b negative (for -J^T * r)
    for i in 0..b.nrows() {
        b[(i, 0)] = -b[(i, 0)];
    }

    // Create a copy of augmented for in-place operations
    let mut a = augmented.clone();

    // Simple Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_val = a[(i, i)].abs();
        let mut max_row = i;
        for j in i + 1..n {
            let val_abs = a[(j, i)].abs();
            if val_abs > max_val {
                max_val = val_abs;
                max_row = j;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in i..n {
                let temp = a[(i, j)];
                a[(i, j)] = a[(max_row, j)];
                a[(max_row, j)] = temp;
            }
            let temp = b[(i, 0)];
            b[(i, 0)] = b[(max_row, 0)];
            b[(max_row, 0)] = temp;
        }

        // Eliminate below
        for j in i + 1..n {
            let factor = a[(j, i)] / a[(i, i)];
            for k in i..n {
                a[(j, k)] = a[(j, k)] - factor * a[(i, k)];
            }
            b[(j, 0)] = b[(j, 0)] - factor * b[(i, 0)];
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for j in i + 1..n {
            sum += a[(i, j)] * step[(j, 0)];
        }
        step[(i, 0)] = (b[(i, 0)] - sum) / a[(i, i)];
    }

    // Calculate step norm and predicted reduction
    // Compute ||diag * step||
    let mut diag_step = Mat::zeros(n, 1);
    for i in 0..n {
        diag_step[(i, 0)] = diag[(i, 0)] * step[(i, 0)];
    }
    // Calculate Euclidean norm manually
    let mut sum = T::zero();
    for i in 0..diag_step.nrows() {
        sum += diag_step[(i, 0)] * diag_step[(i, 0)];
    }
    let step_norm = sum.sqrt();

    // Compute predicted reduction:
    // 0.5 * (||r||^2 - ||r + J*step||^2)
    // Approximated as: 0.5 * (||r||^2 - ||r + J*step||^2) ≈ -step^T * (J^T*r + 0.5*lambda*diag^2*step)
    let half = T::from_f64(0.5).unwrap();
    let mut lambda_diag_step = Mat::zeros(n, 1);
    for i in 0..n {
        lambda_diag_step[(i, 0)] = lambda * diag[(i, 0)] * diag[(i, 0)] * step[(i, 0)];
    }

    let mut temp = jtr.clone();
    for i in 0..n {
        temp[(i, 0)] += half * lambda_diag_step[(i, 0)];
    }

    let mut predicted_reduction = T::zero();
    for i in 0..n {
        predicted_reduction += -step[(i, 0)] * temp[(i, 0)];
    }

    Ok(LMParameterUpdate {
        lambda,
        step,
        step_norm,
        predicted_reduction,
    })
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
