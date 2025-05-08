# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Focus

The primary goal of this project is:

1. Phase 1: Implement the equivalent functionality of the [levenberg-marquardt](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/) crate while using the [faer](https://docs.rs/faer/latest/faer/) linear algebra library internally
2. Phase 2: Add uncertainty calculations and parameter management similar to [lmfit-py](https://lmfit.github.io/lmfit-py/)

For detailed information about the faer library, please refer to the [FAER.md](./FAER.md) file in this repository.

## Compiler Requirements

This project requires the Rust nightly compiler due to:

1. The use of advanced features in matrix computations and numeric algorithms
2. Potential future integration with developing autodiff features in Rust

A rust-toolchain.toml file is included in the repository to automatically select the nightly compiler.

### Automatic Differentiation Support

This project leverages Rust nightly's experimental `std::autodiff` module (powered by Enzyme) for automatic differentiation:

- **std::autodiff**: Nightly-only experimental API that performs automatic differentiation at the LLVM IR level

The `#[autodiff]` attribute macro has the syntax:

```rust
#[autodiff(NAME, MODE, INPUT_ACTIVITIES..., OUTPUT_ACTIVITY)]
```

Where:

- `NAME`: A valid function name for the generated derivative function
- `MODE`: One of `Forward`, `Reverse`, `ForwardFirst`, `ReverseFirst`, or `Reverse(n)` for batch modes
- `INPUT_ACTIVITIES`: Activity type for each input parameter:
  - **Active**: Parameter is active in differentiation, gradient returned by value
  - **Duplicated**: Parameter is active, gradient accumulated in-place via mutable reference
  - **Const**: Parameter treated as constant (no gradient needed)
  - **DuplicatedNoNeed**: Like Duplicated, but original return value isn't needed
- `OUTPUT_ACTIVITY`: Activity for the output (Active or DuplicatedNoNeed)

Example for a basic function f(x,y) = x² + 3y:

```rust
#[autodiff(df, Reverse, Active, Active, Active)]
fn f(x: f32, y: f32) -> f32 {
    x * x + 3.0 * y
}

// Generated df function returns (original_result, d_dx, d_dy)
// df(5.0, 7.0, 1.0) returns (46.0, 10.0, 3.0)
```

Our tiered approach for jacobian calculation:

1. **User-provided custom Jacobian**: Fastest option when provided by the user
2. **Automatic differentiation**:
   - Automatically computes derivatives for Jacobian matrices and gradients
   - Intelligently selects forward or reverse mode based on problem dimensions
   - Eliminates the need for manually coding derivatives
   - Produces exact derivatives without numerical approximation errors
3. **Numerical differentiation**: Falls back to finite difference methods when autodiff is not applicable
   - Configurable step size and algorithm
   - Available as central, forward, or backward difference

## Build/Test Commands

- Build: `cargo build`
- Run: `cargo run`
- Test all: `cargo test`
- Test specific: `cargo test test_name`
- Test specific module: `cargo test module_name`
- Lint: `cargo clippy -- -D warnings`
- Format: `cargo fmt`
- Benchmark: `cargo bench`
- Documentation: `cargo doc --open`

## Development Approach

- **Compatible Interface**: Implement an interface compatible with the levenberg-marquardt crate
- **Pure Rust**: Keep the implementation in pure Rust
- **Matrix Calculations**: Use faer for ALL matrix and numerical operations internally
  - IMPORTANT: Always leverage faer's built-in functions for numerical calculations
  - NEVER implement numerical algorithms yourself when faer provides the functionality
  - faer is highly optimized with SIMD and parallel processing capabilities
- **Matrix Interoperability**: Support both ndarray and faer matrices with efficient conversion
- **Strict TDD**: Always write failing tests first, then implement code to make them pass
  - Every new feature must have corresponding tests
  - All tests must pass before considering a feature complete
  - Target 90%+ test coverage for core functionality
  - Implement both unit tests and integration tests
- **Error Handling**: Use thiserror for defining library error types and anyhow for error context and propagation

## Implementation Plans

### Phase 1: Core LM Implementation with faer

1. Set up project structure and dependencies
2. Implement matrix conversion utilities between ndarray, faer, and nalgebra
3. Create problem definition trait that maintains compatibility with levenberg-marquardt
4. Implement core LM algorithm with basic functionality
5. Add tiered Jacobian calculation strategy:
   - User-provided custom Jacobian
   - Automatic differentiation (using std::autodiff)
   - Numerical differentiation (finite difference methods)
6. Implement trust region algorithms and step size control
7. Add convergence criteria and robust stopping conditions

### Phase 2: Parameter System and Uncertainty Analysis (lmfit-py compatibility)

**Note: Before beginning Phase 2, conduct a comprehensive review of Phase 1 implementation and replan as needed to ensure proper integration.**

1. Implement parameter system with:
   - Named parameters with metadata
   - Parameter bounds (min/max constraints)
   - Parameter groups and linked parameters
2. Add uncertainty calculation capabilities:
   - Covariance matrix estimation
   - Confidence interval calculations
   - Parameter correlation analysis
   - Goodness-of-fit statistics
3. Create model definition system:
   - Abstract model interface
   - Pre-built common models (Gaussian, Lorentzian, etc.)
   - Composite model support
   - Model serialization/deserialization
4. Implement advanced fitting features:
   - Robust fitting with outlier detection
   - Multiple dataset fitting
   - Global optimization methods
   - Constrained optimization

## Phase 1 Components (Core LM Algorithm)

### LeastSquaresProblem Trait

Implement a trait equivalent to the `LeastSquaresProblem` from the levenberg-marquardt crate, with optional jacobian methods:

```rust
pub trait LeastSquaresProblem<T>
where
    T: RealField,
{
    fn residuals(&self, parameters: &faer::Mat<T>) -> faer::Mat<T>;

    // Optional user-provided Jacobian
    fn jacobian(&self, parameters: &faer::Mat<T>) -> Option<faer::Mat<T>> {
        None // Default implementation returns None, triggering auto or numerical differentiation
    }

    // Hint whether to use autodiff or numerical differentiation when no Jacobian is provided
    fn prefer_autodiff(&self) -> bool {
        true // Default to autodiff when available
    }
}
```

### JacobianCalculator Trait

A trait to abstract the different Jacobian calculation methods:

```rust
pub enum JacobianMethod {
    UserProvided,
    AutoDiff,
    NumericalForward,
    NumericalCentral,
    NumericalBackward,
}

pub trait JacobianCalculator<T> {
    fn calculate_jacobian(
        &self,
        problem: &impl LeastSquaresProblem<T>,
        parameters: &faer::Mat<T>
    ) -> faer::Mat<T>;

    fn method_used(&self) -> JacobianMethod;
}
```

### LevenbergMarquardt Algorithm

Implement a struct similar to the levenberg-marquardt crate's `LevenbergMarquardt`:

```rust
pub struct LevenbergMarquardt {
    // Configuration parameters for the algorithm
    pub max_iterations: usize,
    pub epsilon_1: f64, // First convergence tolerance
    pub epsilon_2: f64, // Second convergence tolerance
    pub tau: f64,      // Initial damping factor scale
    pub jacobian_method: JacobianMethod,
    pub numerical_diff_step_size: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            epsilon_1: 1e-10,
            epsilon_2: 1e-10,
            tau: 1e-3,
            jacobian_method: JacobianMethod::AutoDiff,
            numerical_diff_step_size: 1e-6,
        }
    }
}

impl LevenbergMarquardt {
    pub fn minimize<T, P>(&self, problem: P, initial_guess: &faer::Mat<T>) -> MinimizationReport<T>
    where
        T: RealField,
        P: LeastSquaresProblem<T>,
    {
        // Implementation using faer for matrix operations
    }
}
```

### MinimizationReport

Implement a results struct similar to the levenberg-marquardt crate's `MinimizationReport`:

```rust
pub struct MinimizationReport<T>
where
    T: RealField,
{
    pub solution_params: faer::Mat<T>,
    pub residuals: faer::Mat<T>,
    pub objective_function: T,
    pub iterations: usize,
    pub jacobian: Option<faer::Mat<T>>,
    pub jacobian_method_used: JacobianMethod,
    pub success: bool,
    pub termination_reason: TerminationReason,
    pub execution_time: std::time::Duration,
}

pub enum TerminationReason {
    Converged,
    MaxIterationsReached,
    SmallRelativeReduction,
    SmallParameters,
    InvalidJacobian,
    InvalidResiduals,
    Other(String),
}
```

## Phase 2 Components (Parameter System and Uncertainty)

**Note: These are preliminary designs for Phase 2 and should be revised based on the experience and outcomes from Phase 1 implementation.**

### Parameter System

A parameter system inspired by lmfit-py with bounds and constraints:

```rust
pub struct Parameter<T> {
    pub name: String,
    pub value: T,
    pub min: Option<T>,
    pub max: Option<T>,
    pub vary: bool,  // Fixed or variable during fitting
    pub brute_step: Option<T>, // Step size for brute-force grid searches
    pub user_data: HashMap<String, Value>, // Custom metadata
}

pub struct Parameters<T> {
    parameters: HashMap<String, Parameter<T>>,
}

impl<T> Parameters<T>
where
    T: RealField
{
    pub fn add(&mut self, name: &str, value: T) -> &mut Parameter<T>;
    pub fn add_many(&mut self, params: &[(&str, T)]);
    pub fn get(&self, name: &str) -> Option<&Parameter<T>>;
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Parameter<T>>;
    pub fn values(&self) -> HashMap<String, T>;
    pub fn valuesdict(&self) -> HashMap<String, T>;
    pub fn to_matrix(&self) -> faer::Mat<T>; // Get varying parameters as matrix
    pub fn from_matrix(&mut self, values: &faer::Mat<T>); // Update from matrix
    pub fn validate(&self) -> Result<(), ParameterError>;
}
```

### Model System

An abstract model system for defining models and composite models:

```rust
pub trait Model<T>
where
    T: RealField
{
    fn eval(&self, params: &Parameters<T>, x: &faer::Mat<T>) -> faer::Mat<T>;
    fn guess_params(&self, data: &faer::Mat<T>, x: &faer::Mat<T>) -> Parameters<T>;
    fn name(&self) -> &str;
    fn param_names(&self) -> Vec<String>;
    fn set_param_hint(&mut self, name: &str, value: T, vary: bool, min: Option<T>, max: Option<T>);
    fn make_params(&self) -> Parameters<T>;
}

pub struct CompositeModel<T> {
    left: Box<dyn Model<T>>,
    right: Box<dyn Model<T>>,
    operation: ModelOperation,
}

pub enum ModelOperation {
    Add,
    Multiply,
    Subtract,
    Divide,
    Compose,
}

// Implementation of standard models (Gaussian, Lorentzian, etc.)
pub struct GaussianModel<T> { /* ... */ }
pub struct LorentzianModel<T> { /* ... */ }
pub struct ExponentialModel<T> { /* ... */ }
// etc.
```

### Uncertainty Calculator

Tools for calculating uncertainties and confidence intervals:

```rust
pub struct UncertaintyCalculator<T> {
    // Configuration and methods
}

impl<T> UncertaintyCalculator<T>
where
    T: RealField,
{
    pub fn covariance_matrix(&self, problem: &impl LeastSquaresProblem<T>,
                            solution: &MinimizationReport<T>) -> faer::Mat<T>;

    pub fn confidence_intervals(&self, problem: &impl LeastSquaresProblem<T>,
                               solution: &MinimizationReport<T>,
                               params: &Parameters<T>,
                               sigma: T) -> HashMap<String, (T, T)>;

    pub fn parameter_summary(&self, params: &Parameters<T>,
                            covariance: &faer::Mat<T>) -> String;

    pub fn goodness_of_fit(&self, params: &Parameters<T>,
                          residuals: &faer::Mat<T>,
                          num_data_points: usize) -> FitStatistics<T>;
}

pub struct FitStatistics<T> {
    pub reduced_chi_square: T,
    pub chi_square: T,
    pub aic: T,
    pub bic: T,
    pub degrees_of_freedom: usize,
}
```

## Matrix Operations with faer

The implementation MUST use faer internally for ALL matrix and numerical operations, while maintaining similar methods with the levenberg-marquardt crate. For comprehensive details on faer's capabilities, API, and usage patterns, refer to the [FAER.md](./FAER.md) file.

**CRITICAL PRIORITY**: Leverage faer's existing implementations rather than creating custom numerical code. faer is highly optimized with SIMD, parallelization, and cache-efficient algorithms that will outperform custom implementations.

Key faer operations to use:

- Matrix decompositions (QR, Cholesky, LU, SVD, etc.)
- Matrix multiplication and inverse operations
- Vector and matrix norms and distance calculations
- Linear system solvers and least squares methods
- Matrix factorization and transformation utilities
- Memory-optimized computational routines

## Technical Specifications

### Phase 1: Levenberg-Marquardt Algorithm Implementation

- Trust region variant of the Levenberg-Marquardt algorithm (matching levenberg-marquardt crate)
- Multiple options for linear algebra decompositions (QR, SVD, Cholesky)
- Step size control with adaptive damping parameter
- Configurable convergence criteria
- Robust fit statistics and diagnostics
- Multiple Jacobian calculation strategies:
  - User-provided analytical Jacobian
  - Automatic differentiation with std::autodiff
  - Numerical differentiation with configurable strategies

### Phase 2: Parameter System and Uncertainty

- Named parameter system with bounds and constraints
- Covariance matrix estimation from Jacobian
- Confidence interval calculation
- Monte Carlo methods for complex uncertainty propagation
- Models as first-class abstractions
- Built-in models for common fitting functions
- Composite model support with operand overloading

### Performance Optimization

- **Maximize faer usage**: Always use faer's built-in functions rather than implementing custom algorithms
  - Leverage faer's SIMD acceleration and parallelism features
  - Use faer's matrix decompositions and solvers directly
  - Prefer high-level faer APIs when available
- **Avoid reimplementation**: Never reimplement numerical algorithms that faer already provides
  - This includes matrix operations, decompositions, solvers, and numerical utilities
  - Custom implementations are almost always less efficient than faer's optimized code
- **Memory management**: Use faer's memory optimization strategies for large problems
  - Utilize faer's `MemStack` for temporary allocations
  - Reuse memory where possible for multiple operations
- **Algorithm selection**: Choose optimal decomposition methods based on problem characteristics
  - Follow faer's recommendations for specific problem types
- **Autodiff integration**: Use automatic differentiation when possible for better accuracy
  - Combine faer with Rust's autodiff capabilities

## Directory Structure

### Phase 1: Core Infrastructure

- **src/**
  - **lib.rs** - Main library entry point and module declarations
  - **error.rs** - Library error definitions using thiserror and anyhow
  - **problem.rs** - Problem definition trait and implementations
  - **lm/** - Main Levenberg-Marquardt algorithm implementation
    - **algorithm.rs** - Core algorithm steps
    - **trust_region.rs** - Trust region implementation
    - **convergence.rs** - Convergence criteria
    - **step.rs** - Step calculation and damping
    - **jacobian.rs** - Different Jacobian calculation strategies
  - **utils/** - Common utilities and helper functions
    - **finite_difference.rs** - Numerical differentiation utilities
    - **matrix_convert.rs** - Conversion utilities between matrix types
    - **autodiff.rs** - Automatic differentiation utilities
- **tests/**
  - **matrix_conversion.rs** - Tests for matrix conversion utilities
  - **problem_definition.rs** - Tests for problem definition implementation
  - **lm/** - Tests for Levenberg-Marquardt implementation
    - **algorithm.rs** - Tests for core algorithm
    - **trust_region.rs** - Tests for trust region implementation
    - **convergence.rs** - Tests for convergence criteria
    - **jacobian.rs** - Tests for Jacobian calculation methods
  - **integration/** - Integration tests
    - **levenberg_marquardt_comparison.rs** - Comparison with levenberg-marquardt crate results
- **examples/**
  - **basic_fitting.rs** - Simple fitting examples
  - **levenberg_marquardt_compatibility.rs** - Examples showing compatibility with levenberg-marquardt crate
  - **jacobian_methods.rs** - Examples using different Jacobian calculation methods

### Phase 2: Parameter System and Uncertainty

**Note: This directory structure for Phase 2 is tentative and should be revised before beginning Phase 2 implementation.**

- **src/**
  - **parameters/** - Parameter system implementation
    - **parameter.rs** - Individual parameter implementation
    - **parameters.rs** - Collection of parameters
    - **bounds.rs** - Parameter bounds implementation
    - **constraints.rs** - Parameter constraints implementation
  - **models/** - Model system implementation
    - **model.rs** - Base model trait and implementation
    - **composite.rs** - Composite model implementation
    - **peak.rs** - Peak models (Gaussian, Lorentzian, etc.)
    - **step.rs** - Step models
    - **exponential.rs** - Exponential models
    - **polynomial.rs** - Polynomial models
  - **uncertainty/** - Uncertainty calculation implementation
    - **covariance.rs** - Covariance matrix estimation
    - **confidence.rs** - Confidence interval calculations
    - **monte_carlo.rs** - Monte Carlo uncertainty propagation
    - **statistics.rs** - Goodness-of-fit statistics
  - **global_opt/** - Global optimization methods
    - **basin_hopping.rs** - Basin hopping algorithm
    - **differential_evolution.rs** - Differential evolution algorithm
    - **simulated_annealing.rs** - Simulated annealing algorithm
- **tests/**
  - **parameters/** - Tests for parameter system
    - **parameter.rs** - Tests for individual parameters
    - **parameters.rs** - Tests for parameter collections
    - **bounds.rs** - Tests for parameter bounds
    - **constraints.rs** - Tests for parameter constraints
  - **models/** - Tests for model system
    - **model.rs** - Tests for base model
    - **composite.rs** - Tests for composite models
    - **peak.rs** - Tests for peak models
    - **step.rs** - Tests for step models
  - **uncertainty/** - Tests for uncertainty calculations
    - **covariance.rs** - Tests for covariance matrix estimation
    - **confidence.rs** - Tests for confidence interval calculations
    - **monte_carlo.rs** - Tests for Monte Carlo uncertainty propagation
  - **global_opt/** - Tests for global optimization methods
    - **basin_hopping.rs** - Tests for basin hopping algorithm
    - **differential_evolution.rs** - Tests for differential evolution algorithm
    - **simulated_annealing.rs** - Tests for simulated annealing algorithm
  - **integration/** - Integration tests
    - **lmfit_py_comparison.rs** - Comparison with lmfit-py results
- **examples/**
  - **parameter_system.rs** - Examples using the parameter system
  - **parameter_constraints.rs** - Examples using parameter constraints
  - **models.rs** - Examples using built-in models
  - **composite_models.rs** - Examples using composite models
  - **uncertainty_calculation.rs** - Examples calculating uncertainties
  - **global_optimization.rs** - Examples using global optimization methods

## Code Style Guidelines

- **Formatting**: Follow Rust standard formatting with `cargo fmt`
- **Naming**:
  - Use descriptive names for functions and variables
  - Follow Rust naming conventions (snake_case for functions and variables, CamelCase for types)
  - For mathematical formulas, include comments with the standard equation notation
- **Organization**:
  - Split code into modules matching the original levenberg-marquardt implementation while improving organization
  - Place internal utility functions in a dedicated `utils.rs` file
  - Define all error types in a dedicated `error.rs` file
- **Documentation**: All public APIs must have rustdoc comments with mathematical explanation
- **Error Types**: Define specific error types with thiserror and use anyhow for error context and propagation
- **Performance**: Optimize numerically intensive operations, use faer efficiently
- **Testing**:
  - Unit test mathematical correctness against reference values from the levenberg-marquardt crate
  - Place small unit tests in each source file under `#[cfg(test)]` modules
  - Place larger integration tests in the `tests/` directory organized by module

## Project Documentation

The project includes the following key documentation files:

1. **CLAUDE.md** (this file): Provides overall guidance for the project, including implementation details, coding standards, and architectural decisions.
2. **FAER.md**: Contains detailed information about the faer linear algebra library, including usage examples, best practices, and integration patterns that should be followed when implementing this project.
3. **README.md**: General project overview and usage instructions for users of the library.

## Licensing

- This project is MIT-licensed
- Properly attribute and reference the levenberg-marquardt crate
- Ensure all code follows the license requirements

