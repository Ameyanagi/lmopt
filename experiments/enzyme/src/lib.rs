#![feature(autodiff)]

use std::autodiff::autodiff_forward;

/// A monomorphic, allocation-free residual kernel is the boundary that Enzyme
/// can currently differentiate reliably. The main lmopt trait uses `dyn Trait`
/// internally, which std::autodiff explicitly does not yet support.
#[autodiff_forward(rosenbrock_jvp, Dual, Dual)]
pub fn rosenbrock_residuals(parameters: &[f64; 2], residuals: &mut [f64; 2]) {
    residuals[0] = 1.0 - parameters[0];
    residuals[1] = 10.0 * (parameters[1] - parameters[0] * parameters[0]);
}

pub fn enzyme_jacobian(parameters: &[f64; 2]) -> [[f64; 2]; 2] {
    let mut jacobian = [[0.0; 2]; 2];

    for column in 0..2 {
        let mut seed = [0.0; 2];
        seed[column] = 1.0;
        let mut residuals = [0.0; 2];
        let mut tangent_residuals = [0.0; 2];
        rosenbrock_jvp(parameters, &seed, &mut residuals, &mut tangent_residuals);
        for row in 0..2 {
            jacobian[row][column] = tangent_residuals[row];
        }
    }

    jacobian
}

pub fn analytical_jacobian(parameters: &[f64; 2]) -> [[f64; 2]; 2] {
    [[-1.0, 0.0], [-20.0 * parameters[0], 10.0]]
}

pub fn central_difference_jacobian(parameters: &[f64; 2]) -> [[f64; 2]; 2] {
    let mut jacobian = [[0.0; 2]; 2];
    let step = 1e-6;
    for column in 0..2 {
        let mut forward_parameters = *parameters;
        let mut backward_parameters = *parameters;
        forward_parameters[column] += step;
        backward_parameters[column] -= step;
        let mut forward = [0.0; 2];
        let mut backward = [0.0; 2];
        rosenbrock_residuals(&forward_parameters, &mut forward);
        rosenbrock_residuals(&backward_parameters, &mut backward);
        for row in 0..2 {
            jacobian[row][column] = (forward[row] - backward[row]) / (2.0 * step);
        }
    }
    jacobian
}
