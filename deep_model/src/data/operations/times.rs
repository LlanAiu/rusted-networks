// builtin

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::Data;

pub struct DataTimes;

impl DataTimes {
    fn warn_vectors(vector1: &Array1<f32>, vector2: &Array1<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [TIMES]: {:?} and {:?}",
            vector1.dim(),
            vector2.dim()
        );
    }

    fn warn_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [TIMES]: {:?} and {:?}",
            matrix1.dim(),
            matrix2.dim()
        );
    }

    pub fn multiply_scalars(scalar1: &f32, scalar2: &f32) -> Data {
        Data::ScalarF32(scalar1 * scalar2)
    }

    pub fn multiply_vectors(vector1: &Array1<f32>, vector2: &Array1<f32>) -> Data {
        if vector1.dim() == vector2.dim() {
            return Data::VectorF32(vector1 * vector2);
        }
        DataTimes::warn_vectors(vector1, vector2);
        Data::None
    }

    pub fn multiply_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) -> Data {
        if matrix1.dim() == matrix2.dim() {
            return Data::MatrixF32(matrix1 * matrix2);
        }
        DataTimes::warn_matrices(matrix1, matrix2);
        Data::None
    }

    pub fn multiply_scalar_vector(scalar: &f32, vector: &Array1<f32>) -> Data {
        Data::VectorF32(vector * *scalar)
    }

    pub fn multiply_scalar_matrix(scalar: &f32, matrix: &Array2<f32>) -> Data {
        Data::MatrixF32(matrix * *scalar)
    }
}
