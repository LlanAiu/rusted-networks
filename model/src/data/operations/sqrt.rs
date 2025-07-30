// builtin

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::Data;

pub struct DataSquareRoot;

impl DataSquareRoot {
    pub fn square_root_scalar(scalar: &f32) -> Data {
        Data::ScalarF32(f32::sqrt(*scalar))
    }

    pub fn square_root_vector(vector: &Array1<f32>) -> Data {
        let mut vec_copy = vector.clone();
        vec_copy.mapv_inplace(f32::sqrt);
        Data::VectorF32(vec_copy)
    }

    pub fn square_root_matrix(matrix: &Array2<f32>) -> Data {
        let mut matrix_copy = matrix.clone();
        matrix_copy.mapv_inplace(f32::sqrt);
        return Data::MatrixF32(matrix_copy);
    }
}
