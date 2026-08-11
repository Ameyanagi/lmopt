pub mod finite_difference;
pub mod jacobian;
#[cfg(any(feature = "nalgebra", feature = "ndarray"))]
pub mod matrix_convert;
