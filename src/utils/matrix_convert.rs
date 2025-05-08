use crate::Result;
use faer::Mat;
use faer_traits::RealField;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};

/// Trait for converting matrices from different libraries to faer matrices
pub trait IntoFaer<T> {
    /// Convert to a faer matrix
    fn into_faer(self) -> Result<Mat<T>>;
}

/// Trait for converting faer matrices to other matrix formats
pub trait FromFaer<T> {
    /// Convert from a faer matrix
    fn from_faer(mat: &Mat<T>) -> Result<Self>
    where
        Self: Sized;
}

// Implementation for ndarray -> faer
impl<T: RealField + Copy> IntoFaer<T> for &Array2<T> {
    fn into_faer(self) -> Result<Mat<T>> {
        let rows = self.nrows();
        let cols = self.ncols();
        let mut result = Mat::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result[(i, j)] = self[[i, j]];
            }
        }

        Ok(result)
    }
}

// Implementation for ndarray vector -> faer
impl<T: RealField + Copy> IntoFaer<T> for &Array1<T> {
    fn into_faer(self) -> Result<Mat<T>> {
        let rows = self.len();
        let mut result = Mat::zeros(rows, 1);

        for i in 0..rows {
            result[(i, 0)] = self[i];
        }

        Ok(result)
    }
}

// Implementation for nalgebra -> faer
impl<T: RealField + Copy> IntoFaer<T> for &DMatrix<T> {
    fn into_faer(self) -> Result<Mat<T>> {
        let rows = self.nrows();
        let cols = self.ncols();
        let mut result = Mat::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result[(i, j)] = self[(i, j)];
            }
        }

        Ok(result)
    }
}

// Implementation for nalgebra vector -> faer
impl<T: RealField + Copy> IntoFaer<T> for &DVector<T> {
    fn into_faer(self) -> Result<Mat<T>> {
        let rows = self.nrows();
        let mut result = Mat::zeros(rows, 1);

        for i in 0..rows {
            result[(i, 0)] = self[i];
        }

        Ok(result)
    }
}

// Implementation for faer -> ndarray
impl<T: RealField + Copy> FromFaer<T> for Array2<T> {
    fn from_faer(mat: &Mat<T>) -> Result<Self> {
        let rows = mat.nrows();
        let cols = mat.ncols();
        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                result[[i, j]] = mat[(i, j)];
            }
        }

        Ok(result)
    }
}

// Implementation for faer -> ndarray vector (from column vector)
impl<T: RealField + Copy> FromFaer<T> for Array1<T> {
    fn from_faer(mat: &Mat<T>) -> Result<Self> {
        let rows = mat.nrows();
        let cols = mat.ncols();

        // Check if the matrix is a column vector
        if cols != 1 {
            return Err(crate::Error::DimensionMismatch("Cannot convert matrix to Array1: not a column vector".to_string()).into());
        }

        let mut result = Array1::zeros(rows);

        for i in 0..rows {
            result[i] = mat[(i, 0)];
        }

        Ok(result)
    }
}

// Implementation for faer -> nalgebra
impl<T: RealField + Copy + 'static> FromFaer<T> for DMatrix<T> {
    fn from_faer(mat: &Mat<T>) -> Result<Self> {
        let rows = mat.nrows();
        let cols = mat.ncols();
        let mut result = DMatrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result[(i, j)] = mat[(i, j)];
            }
        }

        Ok(result)
    }
}

// Implementation for faer -> nalgebra vector (from column vector)
impl<T: RealField + Copy + 'static> FromFaer<T> for DVector<T> {
    fn from_faer(mat: &Mat<T>) -> Result<Self> {
        let rows = mat.nrows();
        let cols = mat.ncols();

        // Check if the matrix is a column vector
        if cols != 1 {
            return Err(crate::Error::DimensionMismatch("Cannot convert matrix to DVector: not a column vector".to_string()).into());
        }

        let mut result = DVector::zeros(rows);

        for i in 0..rows {
            result[i] = mat[(i, 0)];
        }

        Ok(result)
    }
}
