use approx::assert_relative_eq;
use faer::Mat;
use lmopt::utils::jacobian::{get_jacobian_calculator, JacobianCalculator};
use lmopt::{JacobianMethod, LeastSquaresProblem, Result};

// Test problem with nonlinear functions that have known analytical derivatives
struct NonlinearProblem;

impl LeastSquaresProblem<f64> for NonlinearProblem {
    fn residuals(&self, parameters: &Mat<f64>) -> Result<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Residuals that combine polynomial and transcendental functions
        let mut residuals = Mat::zeros(4, 1);
        residuals[(0, 0)] = x * x + y; // r1 = x^2 + y
        residuals[(1, 0)] = x * x * y; // r2 = x^2 * y
        residuals[(2, 0)] = x.sin() + y.cos(); // r3 = sin(x) + cos(y)
        residuals[(3, 0)] = x.exp() * y; // r4 = exp(x) * y

        Ok(residuals)
    }

    // Analytical Jacobian for comparison
    fn jacobian(&self, parameters: &Mat<f64>) -> Option<Mat<f64>> {
        let x = parameters[(0, 0)];
        let y = parameters[(1, 0)];

        // Jacobian matrix:
        // ∂r1/∂x = 2x      ∂r1/∂y = 1
        // ∂r2/∂x = 2xy     ∂r2/∂y = x^2
        // ∂r3/∂x = cos(x)  ∂r3/∂y = -sin(y)
        // ∂r4/∂x = exp(x)*y ∂r4/∂y = exp(x)
        let mut jacobian = Mat::zeros(4, 2);

        // First residual: x^2 + y
        jacobian[(0, 0)] = 2.0 * x;
        jacobian[(0, 1)] = 1.0;

        // Second residual: x^2 * y
        jacobian[(1, 0)] = 2.0 * x * y;
        jacobian[(1, 1)] = x * x;

        // Third residual: sin(x) + cos(y)
        jacobian[(2, 0)] = x.cos();
        jacobian[(2, 1)] = -y.sin();

        // Fourth residual: exp(x) * y
        jacobian[(3, 0)] = x.exp() * y;
        jacobian[(3, 1)] = x.exp();

        Some(jacobian)
    }
}

#[test]
#[cfg(feature = "autodiff")]
fn test_compare_numerical_with_autodiff() {
    let problem = NonlinearProblem;

    // Test with multiple parameter values to ensure robust comparison
    let test_points = vec![
        // (x, y)
        (1.0, 2.0),
        (0.5, -1.5),
        (2.0, 0.1),
        (-0.7, 1.3),
        (0.0, 0.0), // Special case: origin
    ];

    for (i, (x, y)) in test_points.iter().enumerate() {
        let params = Mat::from_fn(2, 1, |j, _| if j == 0 { *x } else { *y });

        // Get the analytical Jacobian (our reference)
        let analytical = problem.jacobian(&params).unwrap();

        // Calculate Jacobian using autodiff
        let autodiff_calculator = get_jacobian_calculator::<f64>(JacobianMethod::AutoDiff, 0.0);
        let autodiff_jacobian = autodiff_calculator.calculate_jacobian(&problem, &params).unwrap();

        // Calculate Jacobian using numerical central differences
        let numerical_calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalCentral, 1e-6);
        let numerical_jacobian = numerical_calculator.calculate_jacobian(&problem, &params).unwrap();

        // Now compare the three Jacobians
        for row in 0..4 {
            for col in 0..2 {
                // Compare autodiff vs analytical (should be very close)
                let autodiff_error = (autodiff_jacobian[(row, col)] - analytical[(row, col)]).abs();

                // Compare numerical vs analytical
                let numerical_error = (numerical_jacobian[(row, col)] - analytical[(row, col)]).abs();

                // Compare autodiff vs numerical (should be within reasonable range)
                let diff = (autodiff_jacobian[(row, col)] - numerical_jacobian[(row, col)]).abs();

                // Print comparison information for debugging
                println!(
                    "Test point {}, element ({},{}): analytical={:.8}, autodiff={:.8} (err={:.8}), numerical={:.8} (err={:.8}), diff={:.8}",
                    i,
                    row,
                    col,
                    analytical[(row, col)],
                    autodiff_jacobian[(row, col)],
                    autodiff_error,
                    numerical_jacobian[(row, col)],
                    numerical_error,
                    diff
                );

                // Reasonable error tolerance for comparison
                let tolerance = 1e-6;

                // Assert that autodiff and numerical are close to each other
                assert_relative_eq!(autodiff_jacobian[(row, col)], numerical_jacobian[(row, col)], max_relative = tolerance, epsilon = tolerance);

                // Also check both are close to analytical
                assert_relative_eq!(autodiff_jacobian[(row, col)], analytical[(row, col)], max_relative = tolerance, epsilon = tolerance);

                assert_relative_eq!(numerical_jacobian[(row, col)], analytical[(row, col)], max_relative = tolerance, epsilon = tolerance);
            }
        }
    }
}

#[test]
fn test_numerical_methods_comparison() {
    let problem = NonlinearProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 });

    // Get the analytical Jacobian (reference)
    let analytical = problem.jacobian(&params).unwrap();

    // Calculate Jacobian using different numerical methods
    let forward_calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalForward, 1e-6);
    let central_calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalCentral, 1e-6);
    let backward_calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalBackward, 1e-6);

    let forward_jacobian = forward_calculator.calculate_jacobian(&problem, &params).unwrap();
    let central_jacobian = central_calculator.calculate_jacobian(&problem, &params).unwrap();
    let backward_jacobian = backward_calculator.calculate_jacobian(&problem, &params).unwrap();

    // Compare accuracy of different numerical methods
    for row in 0..4 {
        for col in 0..2 {
            let forward_error = (forward_jacobian[(row, col)] - analytical[(row, col)]).abs();
            let central_error = (central_jacobian[(row, col)] - analytical[(row, col)]).abs();
            let backward_error = (backward_jacobian[(row, col)] - analytical[(row, col)]).abs();

            println!(
                "Element ({},{}): analytical={:.8}, forward_err={:.8}, central_err={:.8}, backward_err={:.8}",
                row,
                col,
                analytical[(row, col)],
                forward_error,
                central_error,
                backward_error
            );

            // Central difference should generally be more accurate than forward or backward
            // However, this isn't always true depending on function behavior at specific points
            // So we just verify all are reasonably close to analytical
            assert!(forward_error < 1e-5, "Forward difference error too large at ({},{})!", row, col);
            assert!(central_error < 1e-6, "Central difference error too large at ({},{})!", row, col);
            assert!(backward_error < 1e-5, "Backward difference error too large at ({},{})!", row, col);

            // Central difference typically has higher precision
            assert_relative_eq!(central_jacobian[(row, col)], analytical[(row, col)], max_relative = 1e-6, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_step_size_impact() {
    let problem = NonlinearProblem;
    let params = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 });

    // Get the analytical Jacobian (reference)
    let analytical = problem.jacobian(&params).unwrap();

    // Test different step sizes
    let step_sizes = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];

    for step_size in step_sizes.iter() {
        let calculator = get_jacobian_calculator::<f64>(JacobianMethod::NumericalCentral, *step_size);
        let jacobian = calculator.calculate_jacobian(&problem, &params).unwrap();

        // Calculate maximum error across all elements
        let mut max_error = 0.0;
        for row in 0..4 {
            for col in 0..2 {
                let error = (jacobian[(row, col)] - analytical[(row, col)]).abs();
                max_error = max_error.max(error);
            }
        }

        println!("Step size: {:.1e}, maximum error: {:.8e}", step_size, max_error);

        // For medium step sizes (1e-6 to 1e-8), error should be relatively small
        if *step_size >= 1e-8 && *step_size <= 1e-6 {
            assert!(max_error < 1e-5, "Error too large with step size {}", step_size);
        }
    }
}
