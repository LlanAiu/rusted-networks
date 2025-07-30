// builtin

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::Data;

pub struct DataElementSum;

impl DataElementSum {
    pub fn element_sum_scalar(scalar: &f32) -> Data {
        Data::ScalarF32(*scalar)
    }

    pub fn element_sum_vector(vector: &Array1<f32>) -> Data {
        Data::ScalarF32(vector.sum())
    }

    pub fn element_sum_matrix(matrix: &Array2<f32>) -> Data {
        Data::ScalarF32(matrix.sum())
    }
}
