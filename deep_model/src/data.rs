// builtin

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::operations::{
    matmul::DataMatMul, minus::DataMinus, plus::DataPlus, times::DataTimes,
};
pub mod data_container;
pub mod operations;

#[derive(Debug, Clone)]
pub enum Data {
    ScalarF32(f32),
    VectorF32(Array1<f32>),
    MatrixF32(Array2<f32>),
    None,
}

impl Data {
    pub fn zero() -> Data {
        Data::ScalarF32(0.0)
    }

    pub fn one() -> Data {
        Data::ScalarF32(1.0)
    }

    fn warn_operation(this: &Data, other: &Data, operation: &str) {
        let this_type = this.variant_name();
        let other_type = other.variant_name();
        println!(
            "Data::None returned on unsupported data type pair for operation [{operation}]: {this_type} and {other_type}!"
        );
    }
}

impl Data {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Data::ScalarF32(_) => "ScalarF32",
            Data::VectorF32(_) => "VectorF32",
            Data::MatrixF32(_) => "MatrixF32",
            Data::None => "None",
        }
    }

    pub fn plus(&self, other: &Data) -> Data {
        match (self, other) {
            (Data::ScalarF32(scalar1), Data::ScalarF32(scalar2)) => {
                DataPlus::sum_scalars(scalar1, scalar2)
            }
            (Data::ScalarF32(scalar), Data::VectorF32(vector)) => {
                DataPlus::sum_scalar_vector(scalar, vector)
            }
            (Data::ScalarF32(scalar), Data::MatrixF32(matrix)) => {
                DataPlus::sum_scalar_matrix(scalar, matrix)
            }
            (Data::VectorF32(vector), Data::ScalarF32(scalar)) => {
                DataPlus::sum_scalar_vector(scalar, vector)
            }
            (Data::VectorF32(vector1), Data::VectorF32(vector2)) => {
                DataPlus::sum_vectors(vector1, vector2)
            }
            (Data::MatrixF32(matrix), Data::ScalarF32(scalar)) => {
                DataPlus::sum_scalar_matrix(scalar, matrix)
            }
            (Data::MatrixF32(matrix1), Data::MatrixF32(matrix2)) => {
                DataPlus::sum_matrices(matrix1, matrix2)
            }
            _ => {
                Data::warn_operation(self, other, "PLUS");
                Data::None
            }
        }
    }

    pub fn minus(&self, other: &Data) -> Data {
        match (self, other) {
            (Data::ScalarF32(scalar1), Data::ScalarF32(scalar2)) => {
                DataMinus::subtract_scalars(scalar1, scalar2)
            }
            (Data::ScalarF32(scalar), Data::VectorF32(vector)) => {
                DataMinus::subtract_scalar_vector(scalar, vector)
            }
            (Data::ScalarF32(scalar), Data::MatrixF32(matrix)) => {
                DataMinus::subtract_scalar_matrix(scalar, matrix)
            }
            (Data::VectorF32(vector), Data::ScalarF32(scalar)) => {
                DataMinus::subtract_vector_scalar(vector, scalar)
            }
            (Data::VectorF32(vector1), Data::VectorF32(vector2)) => {
                DataMinus::subtract_vectors(vector1, vector2)
            }
            (Data::MatrixF32(matrix), Data::ScalarF32(scalar)) => {
                DataMinus::subtract_matrix_scalar(matrix, scalar)
            }
            (Data::MatrixF32(matrix1), Data::MatrixF32(matrix2)) => {
                DataMinus::subtract_matrices(matrix1, matrix2)
            }
            _ => {
                Data::warn_operation(self, other, "MINUS");
                Data::None
            }
        }
    }

    pub fn times(&self, other: &Data) -> Data {
        match (self, other) {
            (Data::ScalarF32(scalar1), Data::ScalarF32(scalar2)) => {
                DataTimes::multiply_scalars(scalar1, scalar2)
            }
            (Data::ScalarF32(scalar), Data::VectorF32(vector)) => {
                DataTimes::multiply_scalar_vector(scalar, vector)
            }
            (Data::ScalarF32(scalar), Data::MatrixF32(matrix)) => {
                DataTimes::multiply_scalar_matrix(scalar, matrix)
            }
            (Data::VectorF32(vector), Data::ScalarF32(scalar)) => {
                DataTimes::multiply_scalar_vector(scalar, vector)
            }
            (Data::VectorF32(vector1), Data::VectorF32(vector2)) => {
                DataTimes::multiply_vectors(vector1, vector2)
            }
            (Data::MatrixF32(matrix), Data::ScalarF32(scalar)) => {
                DataTimes::multiply_scalar_matrix(scalar, matrix)
            }
            (Data::MatrixF32(matrix1), Data::MatrixF32(matrix2)) => {
                DataTimes::multiply_matrices(matrix1, matrix2)
            }
            _ => {
                Data::warn_operation(self, other, "TIMES");
                Data::None
            }
        }
    }

    pub fn matmul(&self, other: &Data) -> Data {
        match (self, other) {
            (Data::VectorF32(vector1), Data::VectorF32(vector2)) => {
                DataMatMul::matmul_vectors(vector1, vector2)
            }
            (Data::VectorF32(vector), Data::MatrixF32(matrix)) => {
                DataMatMul::matmul_vector_matrix(vector, matrix)
            }
            (Data::MatrixF32(matrix), Data::VectorF32(vector)) => {
                DataMatMul::matmul_matrix_vector(matrix, vector)
            }
            (Data::MatrixF32(matrix1), Data::MatrixF32(matrix2)) => {
                DataMatMul::matmul_matrices(matrix1, matrix2)
            }
            _ => {
                Data::warn_operation(self, other, "MATMUL");
                Data::None
            }
        }
    }

    pub fn dim(&self) -> &[usize] {
        match self {
            Data::ScalarF32(_) => &[1],
            Data::VectorF32(vec) => vec.shape(),
            Data::MatrixF32(matrix) => matrix.shape(),
            Data::None => &[],
        }
    }
}

impl Default for Data {
    fn default() -> Self {
        Data::zero()
    }
}
