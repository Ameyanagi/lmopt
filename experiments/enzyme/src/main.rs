use lmopt_enzyme_experiment::{analytical_jacobian, enzyme_jacobian};

fn main() {
    let parameters = [-1.2, 1.0];
    let jacobian = enzyme_jacobian(&parameters);
    let expected = analytical_jacobian(&parameters);

    for row in 0..2 {
        for column in 0..2 {
            assert!((jacobian[row][column] - expected[row][column]).abs() < 1e-12);
        }
    }

    println!("validated Enzyme Jacobian: {jacobian:?}");
}
