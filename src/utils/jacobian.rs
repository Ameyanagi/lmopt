use crate::{lm::JacobianMethod, utils::finite_difference::FiniteDifferenceMethod, Error, LeastSquaresProblem, Result};
use faer::Mat;
use faer_traits::RealField;
use num_traits::FromPrimitive;

/// Trait for calculating Jacobian matrices.
pub trait JacobianCalculator<T: RealField + Copy> {
    /// Calculate the Jacobian matrix for the given problem and parameters.
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>>;

    /// Returns the method used for Jacobian calculation.
    fn method_used(&self) -> JacobianMethod;
}

/// Trait to erase the problem's generic type parameters for dyn dispatch
pub trait EraseTypes<T: RealField + Copy> {
    /// Compute the residuals for the given parameters.
    fn erased_residuals(&self, parameters: &Mat<T>) -> Result<Mat<T>>;

    /// Optionally compute the Jacobian matrix for the given parameters.
    fn erased_jacobian(&self, parameters: &Mat<T>) -> Option<Mat<T>>;
}

// Implement EraseTypes for any type that implements LeastSquaresProblem
impl<T: RealField + Copy, P: LeastSquaresProblem<T>> EraseTypes<T> for P {
    fn erased_residuals(&self, parameters: &Mat<T>) -> Result<Mat<T>> {
        self.residuals(parameters)
    }

    fn erased_jacobian(&self, parameters: &Mat<T>) -> Option<Mat<T>> {
        self.jacobian(parameters)
    }
}

/// Jacobian calculator using user-provided function.
pub struct UserProvidedJacobian;

impl<T: RealField + Copy> JacobianCalculator<T> for UserProvidedJacobian {
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>> {
        match problem.erased_jacobian(parameters) {
            Some(jacobian) => Ok(jacobian),
            None => Err(Error::UserFunction("No user-provided Jacobian".to_string()).into()),
        }
    }

    fn method_used(&self) -> JacobianMethod {
        JacobianMethod::UserProvided
    }
}

/// Jacobian calculator using numerical differentiation.
pub struct NumericalJacobian {
    method: FiniteDifferenceMethod,
    step_size: f64,
}

impl NumericalJacobian {
    /// Create a new numerical Jacobian calculator with the given method and step size.
    pub fn new(method: FiniteDifferenceMethod, step_size: f64) -> Self {
        Self { method, step_size }
    }
}

impl<T: RealField + Copy + num_traits::FromPrimitive> JacobianCalculator<T> for NumericalJacobian {
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>> {
        // Convert the step size
        let step_size = T::from_f64(self.step_size).ok_or_else(|| Error::Numerical("Failed to convert step size".to_string()))?;

        // Calculate using numerical differentiation
        let residuals = problem.erased_residuals(parameters)?;
        let n_params = parameters.nrows();
        let n_residuals = residuals.nrows();

        let mut jacobian = Mat::zeros(n_residuals, n_params);
        let mut perturbed_params = parameters.clone();

        for j in 0..n_params {
            match self.method {
                FiniteDifferenceMethod::Forward => {
                    // Forward difference: (f(x+h) - f(x)) / h
                    let original_val = parameters[(j, 0)];
                    perturbed_params[(j, 0)] = original_val + step_size;

                    let perturbed_residuals = problem.erased_residuals(&perturbed_params)?;

                    // Calculate derivative
                    for i in 0..n_residuals {
                        jacobian[(i, j)] = (perturbed_residuals[(i, 0)] - residuals[(i, 0)]) / step_size;
                    }

                    // Restore original value
                    perturbed_params[(j, 0)] = original_val;
                }
                FiniteDifferenceMethod::Central => {
                    // Central difference: (f(x+h) - f(x-h)) / (2*h)
                    let original_val = parameters[(j, 0)];

                    // f(x+h)
                    perturbed_params[(j, 0)] = original_val + step_size;
                    let forward_residuals = problem.erased_residuals(&perturbed_params)?;

                    // f(x-h)
                    perturbed_params[(j, 0)] = original_val - step_size;
                    let backward_residuals = problem.erased_residuals(&perturbed_params)?;

                    // Calculate derivative
                    for i in 0..n_residuals {
                        jacobian[(i, j)] = (forward_residuals[(i, 0)] - backward_residuals[(i, 0)]) / (step_size + step_size);
                    }

                    // Restore original value
                    perturbed_params[(j, 0)] = original_val;
                }
                FiniteDifferenceMethod::Backward => {
                    // Backward difference: (f(x) - f(x-h)) / h
                    let original_val = parameters[(j, 0)];
                    perturbed_params[(j, 0)] = original_val - step_size;

                    let perturbed_residuals = problem.erased_residuals(&perturbed_params)?;

                    // Calculate derivative
                    for i in 0..n_residuals {
                        jacobian[(i, j)] = (residuals[(i, 0)] - perturbed_residuals[(i, 0)]) / step_size;
                    }

                    // Restore original value
                    perturbed_params[(j, 0)] = original_val;
                }
            }
        }

        Ok(jacobian)
    }

    fn method_used(&self) -> JacobianMethod {
        match self.method {
            FiniteDifferenceMethod::Central => JacobianMethod::NumericalCentral,
            FiniteDifferenceMethod::Forward => JacobianMethod::NumericalForward,
            FiniteDifferenceMethod::Backward => JacobianMethod::NumericalBackward,
        }
    }
}

// Factory function to create appropriate JacobianCalculator based on method
pub fn get_jacobian_calculator<T: RealField + Copy + num_traits::FromPrimitive + 'static>(method: JacobianMethod, step_size: f64) -> Box<dyn JacobianCalculator<T>> {
    match method {
        JacobianMethod::UserProvided => Box::new(UserProvidedJacobian),
        JacobianMethod::NumericalCentral => Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Central, step_size)),
        JacobianMethod::NumericalForward => Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Forward, step_size)),
        JacobianMethod::NumericalBackward => Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Backward, step_size)),
        #[cfg(feature = "autodiff")]
        JacobianMethod::AutoDiff => {
            // Check the concrete type
            let type_name = std::any::type_name::<T>();

            match type_name {
                "f64" => {
                    #[allow(clippy::needless_borrow)]
                    // Use transmute is safe here because we're checking types at runtime
                    let calculator: Box<dyn JacobianCalculator<T>> = if type_name == "f64" {
                        use crate::utils::autodiff::AutoDiffJacobian;
                        // First create a box with the concrete type
                        let boxed: Box<dyn JacobianCalculator<f64>> = Box::new(AutoDiffJacobian);
                        // Then safely transmute to the generic type, which is actually f64
                        unsafe { std::mem::transmute(boxed) }
                    } else if type_name == "f32" {
                        use crate::utils::autodiff::AutoDiffJacobian;
                        // First create a box with the concrete type
                        let boxed: Box<dyn JacobianCalculator<f32>> = Box::new(AutoDiffJacobian);
                        // Then safely transmute to the generic type, which is actually f32
                        unsafe { std::mem::transmute(boxed) }
                    } else {
                        // For other types, fall back to numerical differentiation
                        eprintln!("AutoDiff only supports f32/f64, falling back to numerical differentiation");
                        Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Central, step_size))
                    };

                    calculator
                }
                "f32" => {
                    use crate::utils::autodiff::AutoDiffJacobian;
                    let boxed: Box<dyn JacobianCalculator<f32>> = Box::new(AutoDiffJacobian);
                    unsafe { std::mem::transmute(boxed) }
                }
                _ => {
                    eprintln!("AutoDiff only supports f32/f64, falling back to numerical differentiation for {}", type_name);
                    Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Central, step_size))
                }
            }
        }
        #[cfg(not(feature = "autodiff"))]
        JacobianMethod::AutoDiff => {
            // Fall back to central difference if autodiff is not available
            eprintln!("Warning: AutoDiff was requested but the feature is not enabled. Falling back to numerical differentiation.");
            Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Central, step_size))
        }
    }
}
