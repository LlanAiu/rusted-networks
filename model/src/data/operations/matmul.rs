// builtin

// external
use ndarray::{Array1, Array2, Axis};

// internal
use crate::data::Data;

pub struct DataMatMul;

impl DataMatMul {
    fn warn_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [MATMUL]: {:?} and {:?}",
            matrix1.dim(),
            matrix2.dim()
        );
    }

    fn warn_vector_matrix(vector: &Array1<f32>, matrix: &Array2<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [MATMUL]: {:?} and {:?}",
            vector.dim(),
            matrix.dim()
        );
    }

    fn warn_matrix_vector(matrix: &Array2<f32>, vector: &Array1<f32>) {
        println!(
            "Data::None returned on mismatched dimensions for operation [MATMUL]: {:?} and {:?}",
            matrix.dim(),
            vector.dim()
        );
    }

    pub fn matmul_vectors(vector1: &Array1<f32>, vector2: &Array1<f32>) -> Data {
        let matrix1 = vector1.view().insert_axis(Axis(1));
        let matrix2 = vector2.view().insert_axis(Axis(0));

        let res = matrix1.dot(&matrix2);

        Data::MatrixF32(res)
    }

    pub fn matmul_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) -> Data {
        if matrix1.shape()[1] != matrix2.shape()[0] {
            DataMatMul::warn_matrices(matrix1, matrix2);
            return Data::None;
        }

        let res = matrix1.dot(matrix2);
        Data::MatrixF32(res)
    }

    // Automatic transposition as row vector, use vector multiplication for [n x 1] x [1 x m] effect
    pub fn matmul_vector_matrix(vector: &Array1<f32>, matrix: &Array2<f32>) -> Data {
        if vector.shape()[0] != matrix.shape()[0] {
            DataMatMul::warn_vector_matrix(vector, matrix);
            return Data::None;
        }

        let vector_row = vector.view().insert_axis(Axis(0));
        let res = vector_row.dot(matrix);

        Data::VectorF32(res.remove_axis(Axis(0)))
    }

    pub fn matmul_matrix_vector(matrix: &Array2<f32>, vector: &Array1<f32>) -> Data {
        if matrix.shape()[1] != vector.shape()[0] {
            DataMatMul::warn_matrix_vector(matrix, vector);
            return Data::None;
        }

        let vector_col = vector.view().insert_axis(Axis(1));
        let res = matrix.dot(&vector_col);

        Data::VectorF32(res.remove_axis(Axis(1)))
    }
}
