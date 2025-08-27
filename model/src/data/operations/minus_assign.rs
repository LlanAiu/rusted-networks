// builtin

// external
use ndarray::{Array1, Array2};

// internal

pub struct DataMinusAssign;

impl DataMinusAssign {
    fn warn_vectors(vector1: &Array1<f32>, vector2: &Array1<f32>) {
        println!(
            "Mutation failed on mismatched dimensions for operation [MINUS_INPLACE]: {:?} and {:?}",
            vector1.dim(),
            vector2.dim()
        );
    }

    fn warn_matrices(matrix1: &Array2<f32>, matrix2: &Array2<f32>) {
        println!(
            "Mutation failed on mismatched dimensions for operation [MINUS_INPLACE]: {:?} and {:?}",
            matrix1.dim(),
            matrix2.dim()
        );
    }

    pub fn minus_scalars(l_scalar: &mut f32, r_scalar: &f32) {
        *l_scalar -= r_scalar;
    }

    pub fn minus_vectors(l_vec: &mut Array1<f32>, r_vec: &Array1<f32>) {
        if l_vec.dim() == r_vec.dim() {
            *l_vec -= r_vec;
        } else {
            Self::warn_vectors(&l_vec, r_vec);
        }
    }

    pub fn minus_matrices(l_matrix: &mut Array2<f32>, r_matrix: &Array2<f32>) {
        if l_matrix.dim() == r_matrix.dim() {
            *l_matrix -= r_matrix;
        } else {
            Self::warn_matrices(&l_matrix, r_matrix);
        }
    }

    pub fn minus_vector_scalar(vector: &mut Array1<f32>, scalar: &f32) {
        vector.map_inplace(|f| *f -= scalar);
    }

    pub fn minus_matrix_scalar(matrix: &mut Array2<f32>, scalar: &f32) {
        matrix.map_inplace(|f| *f -= scalar);
    }
}
