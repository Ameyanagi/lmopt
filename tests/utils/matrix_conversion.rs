use faer::Mat;
use lmopt::utils::matrix_convert::{FromFaer, IntoFaer};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};

#[test]
fn test_ndarray_to_faer_conversion() {
    // Create an ndarray matrix
    let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Convert to faer
    let faer_mat = (&arr).into_faer().unwrap();

    // Check dimensions
    assert_eq!(faer_mat.nrows(), 2);
    assert_eq!(faer_mat.ncols(), 3);

    // Check values
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(faer_mat[(i, j)], arr[[i, j]]);
        }
    }
}

#[test]
fn test_ndarray_vector_to_faer_conversion() {
    // Create an ndarray vector
    let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    // Convert to faer
    let faer_mat = (&arr).into_faer().unwrap();

    // Check dimensions
    assert_eq!(faer_mat.nrows(), 3);
    assert_eq!(faer_mat.ncols(), 1);

    // Check values
    for i in 0..3 {
        assert_eq!(faer_mat[(i, 0)], arr[i]);
    }
}

#[test]
fn test_nalgebra_to_faer_conversion() {
    // Create a nalgebra matrix
    let mat = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Convert to faer
    let faer_mat = (&mat).into_faer().unwrap();

    // Check dimensions
    assert_eq!(faer_mat.nrows(), 2);
    assert_eq!(faer_mat.ncols(), 3);

    // Check values
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(faer_mat[(i, j)], mat[(i, j)]);
        }
    }
}

#[test]
fn test_nalgebra_vector_to_faer_conversion() {
    // Create a nalgebra vector
    let vec = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

    // Convert to faer
    let faer_mat = (&vec).into_faer().unwrap();

    // Check dimensions
    assert_eq!(faer_mat.nrows(), 3);
    assert_eq!(faer_mat.ncols(), 1);

    // Check values
    for i in 0..3 {
        assert_eq!(faer_mat[(i, 0)], vec[i]);
    }
}

#[test]
fn test_faer_to_ndarray_conversion() {
    // Create a faer matrix
    let mut faer_mat = Mat::zeros(2, 3);
    for i in 0..2 {
        for j in 0..3 {
            faer_mat[(i, j)] = (i * 3 + j + 1) as f64;
        }
    }

    // Convert to ndarray
    let arr = Array2::from_faer(&faer_mat).unwrap();

    // Check dimensions
    assert_eq!(arr.nrows(), 2);
    assert_eq!(arr.ncols(), 3);

    // Check values
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(arr[[i, j]], faer_mat[(i, j)]);
        }
    }
}

#[test]
fn test_faer_to_ndarray_vector_conversion() {
    // Create a faer column vector
    let mut faer_mat = Mat::zeros(3, 1);
    for i in 0..3 {
        faer_mat[(i, 0)] = (i + 1) as f64;
    }

    // Convert to ndarray vector
    let arr = Array1::from_faer(&faer_mat).unwrap();

    // Check dimensions
    assert_eq!(arr.len(), 3);

    // Check values
    for i in 0..3 {
        assert_eq!(arr[i], faer_mat[(i, 0)]);
    }
}

#[test]
fn test_faer_to_nalgebra_conversion() {
    // Create a faer matrix
    let mut faer_mat = Mat::zeros(2, 3);
    for i in 0..2 {
        for j in 0..3 {
            faer_mat[(i, j)] = (i * 3 + j + 1) as f64;
        }
    }

    // Convert to nalgebra
    let mat = DMatrix::from_faer(&faer_mat).unwrap();

    // Check dimensions
    assert_eq!(mat.nrows(), 2);
    assert_eq!(mat.ncols(), 3);

    // Check values
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(mat[(i, j)], faer_mat[(i, j)]);
        }
    }
}

#[test]
fn test_faer_to_nalgebra_vector_conversion() {
    // Create a faer column vector
    let mut faer_mat = Mat::zeros(3, 1);
    for i in 0..3 {
        faer_mat[(i, 0)] = (i + 1) as f64;
    }

    // Convert to nalgebra vector
    let vec = DVector::from_faer(&faer_mat).unwrap();

    // Check dimensions
    assert_eq!(vec.nrows(), 3);

    // Check values
    for i in 0..3 {
        assert_eq!(vec[i], faer_mat[(i, 0)]);
    }
}

#[test]
fn test_roundtrip_ndarray_conversion() {
    // Create an ndarray matrix
    let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Convert to faer and back
    let faer_mat = (&arr).into_faer().unwrap();
    let arr2 = Array2::from_faer(&faer_mat).unwrap();

    // Check equality
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(arr[[i, j]], arr2[[i, j]]);
        }
    }
}

#[test]
fn test_roundtrip_nalgebra_conversion() {
    // Create a nalgebra matrix
    let mat = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Convert to faer and back
    let faer_mat = (&mat).into_faer().unwrap();
    let mat2 = DMatrix::from_faer(&faer_mat).unwrap();

    // Check equality
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(mat[(i, j)], mat2[(i, j)]);
        }
    }
}
