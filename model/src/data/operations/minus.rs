// builtin

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::Data;

pub struct DataMinus;

impl DataMinus {
    fn warn_vectors(vector1: &Array1<f32>, vector2: &Array1<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [MINUS]: {:?} and {:?}",
            vector1.dim(),
            vector2.dim()
        );
    }

    fn warn_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [MINUS]: {:?} and {:?}",
            matrix1.dim(),
            matrix2.dim()
        );
    }

    pub fn subtract_scalars(scalar1: &f32, scalar2: &f32) -> Data {
        Data::ScalarF32(scalar1 - scalar2)
    }

    pub fn subtract_vectors(vector1: &Array1<f32>, vector2: &Array1<f32>) -> Data {
        if vector1.dim() == vector2.dim() {
            return Data::VectorF32(vector1 - vector2);
        }
        DataMinus::warn_vectors(vector1, vector2);
        Data::None
    }

    pub fn subtract_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) -> Data {
        if matrix1.dim() == matrix2.dim() {
            return Data::MatrixF32(matrix1 - matrix2);
        }
        DataMinus::warn_matrices(matrix1, matrix2);
        Data::None
    }

    pub fn subtract_scalar_vector(scalar: &f32, vector: &Array1<f32>) -> Data {
        Data::VectorF32(vector - *scalar)
    }

    pub fn subtract_vector_scalar(vector: &Array1<f32>, scalar: &f32) -> Data {
        Data::VectorF32(*scalar - vector)
    }

    pub fn subtract_scalar_matrix(scalar: &f32, matrix: &Array2<f32>) -> Data {
        Data::MatrixF32(matrix - *scalar)
    }

    pub fn subtract_matrix_scalar(matrix: &Array2<f32>, scalar: &f32) -> Data {
        Data::MatrixF32(*scalar - matrix)
    }
}
