#[path = "utils/finite_difference.rs"]
mod finite_difference;
#[path = "utils/jacobian.rs"]
mod jacobian;
#[cfg(all(feature = "nalgebra", feature = "ndarray"))]
#[path = "utils/matrix_conversion.rs"]
mod matrix_conversion;
