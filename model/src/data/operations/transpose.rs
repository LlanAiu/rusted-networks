// builtin

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::Data;

pub struct DataTranspose;

impl DataTranspose {
    pub fn transpose_scalar(scalar: &f32) -> Data {
        Data::ScalarF32(*scalar)
    }

    pub fn transpose_vector(vector: &Array1<f32>) -> Data {
        Data::VectorF32(vector.clone())
    }

    pub fn transpose_matrix(matrix: &Array2<f32>) -> Data {
        let transposed = matrix.t().to_owned();
        Data::MatrixF32(transposed)
    }
}
