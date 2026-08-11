use crate::{
    lm::JacobianMethod,
    utils::finite_difference::{calculate_jacobian_with_residuals, FiniteDifferenceMethod},
    Error, LeastSquaresProblem, Result,
};
use faer::Mat;
use faer_traits::RealField;
use num_traits::{Float, FromPrimitive};
use std::cell::Cell;

/// Trait for calculating Jacobian matrices.
pub trait JacobianCalculator<T: RealField + Copy> {
    /// Calculate the Jacobian matrix for the given problem and parameters.
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>>;

    /// Calculate a Jacobian while reusing residuals already evaluated at
    /// `parameters`. Calculators that cannot reuse them may use the default.
    fn calculate_jacobian_with_residuals(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>, _residuals: &Mat<T>) -> Result<Mat<T>> {
        self.calculate_jacobian(problem, parameters)
    }

    /// Returns the method used for Jacobian calculation.
    fn method_used(&self) -> JacobianMethod;
}

/// Trait to erase the problem's generic type parameters for dyn dispatch
pub trait EraseTypes<T: RealField + Copy> {
    /// Compute the residuals for the given parameters.
    fn erased_residuals(&self, parameters: &Mat<T>) -> Result<Mat<T>>;

    /// Optionally compute the Jacobian matrix for the given parameters.
    fn erased_jacobian(&self, parameters: &Mat<T>) -> Result<Option<Mat<T>>>;
}

// Implement EraseTypes for any type that implements LeastSquaresProblem
impl<T: RealField + Copy, P: LeastSquaresProblem<T>> EraseTypes<T> for P {
    fn erased_residuals(&self, parameters: &Mat<T>) -> Result<Mat<T>> {
        self.residuals(parameters)
    }

    fn erased_jacobian(&self, parameters: &Mat<T>) -> Result<Option<Mat<T>>> {
        self.try_jacobian(parameters)
    }
}

/// Jacobian calculator using user-provided function.
pub struct UserProvidedJacobian;

impl<T: RealField + Copy> JacobianCalculator<T> for UserProvidedJacobian {
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>> {
        match problem.erased_jacobian(parameters)? {
            Some(jacobian) => Ok(jacobian),
            None => Err(Error::UserFunction("No user-provided Jacobian".to_string())),
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

impl NumericalJacobian {
    fn calculate_with_residuals<T>(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>, residuals: &Mat<T>) -> Result<Mat<T>>
    where
        T: RealField + Copy + Float + FromPrimitive,
    {
        let relative_step = T::from_f64(self.step_size).ok_or_else(|| Error::Numerical("Failed to convert step size".to_string()))?;
        calculate_jacobian_with_residuals(parameters, residuals, relative_step, self.method, |perturbed| problem.erased_residuals(perturbed))
    }
}

impl<T: RealField + Copy + Float + FromPrimitive> JacobianCalculator<T> for NumericalJacobian {
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>> {
        let residuals = problem.erased_residuals(parameters)?;
        self.calculate_with_residuals(problem, parameters, &residuals)
    }

    fn calculate_jacobian_with_residuals(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>, residuals: &Mat<T>) -> Result<Mat<T>> {
        self.calculate_with_residuals(problem, parameters, residuals)
    }

    fn method_used(&self) -> JacobianMethod {
        match self.method {
            FiniteDifferenceMethod::Central => JacobianMethod::NumericalCentral,
            FiniteDifferenceMethod::Forward => JacobianMethod::NumericalForward,
            FiniteDifferenceMethod::Backward => JacobianMethod::NumericalBackward,
        }
    }
}

/// Automatically chooses an analytical Jacobian when present and otherwise
/// uses central finite differences.
pub struct AutoJacobian {
    numerical: NumericalJacobian,
    method_used: Cell<JacobianMethod>,
}

impl AutoJacobian {
    fn new(step_size: f64) -> Self {
        Self {
            numerical: NumericalJacobian::new(FiniteDifferenceMethod::Central, step_size),
            method_used: Cell::new(JacobianMethod::Auto),
        }
    }
}

impl<T: RealField + Copy + Float + FromPrimitive> JacobianCalculator<T> for AutoJacobian {
    fn calculate_jacobian(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>) -> Result<Mat<T>> {
        if let Some(jacobian) = problem.erased_jacobian(parameters)? {
            self.method_used.set(JacobianMethod::UserProvided);
            Ok(jacobian)
        } else {
            self.method_used.set(JacobianMethod::NumericalCentral);
            self.numerical.calculate_jacobian(problem, parameters)
        }
    }

    fn calculate_jacobian_with_residuals(&self, problem: &dyn EraseTypes<T>, parameters: &Mat<T>, residuals: &Mat<T>) -> Result<Mat<T>> {
        if let Some(jacobian) = problem.erased_jacobian(parameters)? {
            self.method_used.set(JacobianMethod::UserProvided);
            Ok(jacobian)
        } else {
            self.method_used.set(JacobianMethod::NumericalCentral);
            self.numerical.calculate_jacobian_with_residuals(problem, parameters, residuals)
        }
    }

    fn method_used(&self) -> JacobianMethod {
        self.method_used.get()
    }
}

/// Placeholder that fails explicitly rather than silently performing finite
/// differences while claiming they came from automatic differentiation.
pub struct UnavailableAutoDiff;

impl<T: RealField + Copy> JacobianCalculator<T> for UnavailableAutoDiff {
    fn calculate_jacobian(&self, _problem: &dyn EraseTypes<T>, _parameters: &Mat<T>) -> Result<Mat<T>> {
        Err(Error::AutoDiffUnavailable(
            "the Enzyme-backed implementation is not yet available; use JacobianMethod::Auto, provide an analytical Jacobian, or choose a numerical method".to_string(),
        ))
    }

    fn method_used(&self) -> JacobianMethod {
        JacobianMethod::AutoDiff
    }
}

// Factory function to create appropriate JacobianCalculator based on method
pub fn get_jacobian_calculator<T: RealField + Copy + Float + FromPrimitive + 'static>(method: JacobianMethod, step_size: f64) -> Box<dyn JacobianCalculator<T>> {
    match method {
        JacobianMethod::Auto => Box::new(AutoJacobian::new(step_size)),
        JacobianMethod::UserProvided => Box::new(UserProvidedJacobian),
        JacobianMethod::NumericalCentral => Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Central, step_size)),
        JacobianMethod::NumericalForward => Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Forward, step_size)),
        JacobianMethod::NumericalBackward => Box::new(NumericalJacobian::new(FiniteDifferenceMethod::Backward, step_size)),
        JacobianMethod::AutoDiff => Box::new(UnavailableAutoDiff),
    }
}
