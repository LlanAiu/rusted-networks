// builtin

use core::panic;

// external
use ndarray::{Array1, Array2};

// internal
use crate::data::operations::{
    element_sum::DataElementSum, matmul::DataMatMul, minus::DataMinus,
    minus_assign::DataMinusAssign, plus::DataPlus, sqrt::DataSquareRoot, sum_assign::DataSumAssign,
    times::DataTimes, times_assign::DataTimesAssign, transpose::DataTranspose,
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

    pub fn neg_one() -> Data {
        Data::ScalarF32(-1.0)
    }

    pub fn zero_dim(dim: &[usize]) -> Data {
        if dim.len() == 0 {
            return Data::ScalarF32(0.0);
        } else if dim.len() == 1 {
            return Data::VectorF32(Array1::zeros(dim[0]));
        } else if dim.len() == 2 {
            return Data::MatrixF32(Array2::zeros((dim[0], dim[1])));
        }

        panic!("[ZERO_INIT] Unsupported data type dimensions");
    }

    fn warn_operation(this: &Data, other: &Data, operation: &str) {
        let this_type = this.variant_name();
        let other_type = other.variant_name();
        println!(
            "Data::None returned on unsupported data type pair for operation [{operation}]: {this_type} and {other_type}!"
        );
    }

    fn warn_mutate(this_type: &str, other: &Data, operation: &str) {
        let other_type = other.variant_name();
        println!(
            "Mutation failed on unsupported data type pair for operation [{operation}]: {this_type} and {other_type}!"
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

    pub fn sum_assign(&mut self, other: &Data) {
        let variant = self.variant_name();
        match (self, other) {
            (Data::ScalarF32(l_scalar), Data::ScalarF32(r_scalar)) => {
                DataSumAssign::sum_scalars(l_scalar, r_scalar);
            }
            (Data::VectorF32(vector), Data::ScalarF32(scalar)) => {
                DataSumAssign::sum_vector_scalar(vector, scalar);
            }
            (Data::VectorF32(l_vector), Data::VectorF32(r_vector)) => {
                DataSumAssign::sum_vectors(l_vector, r_vector);
            }
            (Data::MatrixF32(matrix), Data::ScalarF32(scalar)) => {
                DataSumAssign::sum_matrix_scalar(matrix, scalar);
            }
            (Data::MatrixF32(l_matrix), Data::MatrixF32(r_matrix)) => {
                DataSumAssign::sum_matrices(l_matrix, r_matrix);
            }
            _ => {
                Data::warn_mutate(variant, other, "PLUS_INPLACE");
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

    pub fn minus_assign(&mut self, other: &Data) {
        let variant = self.variant_name();
        match (self, other) {
            (Data::ScalarF32(l_scalar), Data::ScalarF32(r_scalar)) => {
                DataMinusAssign::minus_scalars(l_scalar, r_scalar);
            }
            (Data::VectorF32(vector), Data::ScalarF32(scalar)) => {
                DataMinusAssign::minus_vector_scalar(vector, scalar);
            }
            (Data::VectorF32(l_vector), Data::VectorF32(r_vector)) => {
                DataMinusAssign::minus_vectors(l_vector, r_vector);
            }
            (Data::MatrixF32(matrix), Data::ScalarF32(scalar)) => {
                DataMinusAssign::minus_matrix_scalar(matrix, scalar);
            }
            (Data::MatrixF32(l_matrix), Data::MatrixF32(r_matrix)) => {
                DataMinusAssign::minus_matrices(l_matrix, r_matrix);
            }
            _ => {
                Data::warn_mutate(variant, other, "MINUS_INPLACE");
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

    pub fn times_assign(&mut self, other: &Data) {
        let variant = self.variant_name();
        match (self, other) {
            (Data::ScalarF32(l_scalar), Data::ScalarF32(r_scalar)) => {
                DataTimesAssign::multiply_scalars(l_scalar, r_scalar);
            }
            (Data::VectorF32(vector), Data::ScalarF32(scalar)) => {
                DataTimesAssign::multiply_vector_scalar(vector, scalar);
            }
            (Data::VectorF32(l_vector), Data::VectorF32(r_vector)) => {
                DataTimesAssign::multiply_vectors(l_vector, r_vector);
            }
            (Data::MatrixF32(matrix), Data::ScalarF32(scalar)) => {
                DataTimesAssign::multiply_matrix_scalar(matrix, scalar);
            }
            (Data::MatrixF32(l_matrix), Data::MatrixF32(r_matrix)) => {
                DataTimesAssign::multiply_matrices(l_matrix, r_matrix);
            }
            _ => {
                Data::warn_mutate(variant, other, "TIMES_INPLACE");
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

    pub fn transpose(&self) -> Data {
        match self {
            Data::ScalarF32(scalar) => DataTranspose::transpose_scalar(scalar),
            Data::VectorF32(vector) => DataTranspose::transpose_vector(vector),
            Data::MatrixF32(matrix) => DataTranspose::transpose_matrix(matrix),
            Data::None => Data::None,
        }
    }

    pub fn element_sum(&self) -> Data {
        match self {
            Data::ScalarF32(scalar) => DataElementSum::element_sum_scalar(scalar),
            Data::VectorF32(vector) => DataElementSum::element_sum_vector(vector),
            Data::MatrixF32(matrix) => DataElementSum::element_sum_matrix(matrix),
            Data::None => Data::None,
        }
    }

    pub fn sqrt(&self) -> Data {
        match self {
            Data::ScalarF32(scalar) => DataSquareRoot::square_root_scalar(scalar),
            Data::VectorF32(vector) => DataSquareRoot::square_root_vector(vector),
            Data::MatrixF32(matrix) => DataSquareRoot::square_root_matrix(matrix),
            Data::None => Data::None,
        }
    }

    pub fn apply_elementwise(&self, func: impl Fn(f32) -> f32) -> Data {
        match self {
            Data::ScalarF32(scalar) => Data::ScalarF32(func(*scalar)),
            Data::VectorF32(vector) => Data::VectorF32(vector.mapv(func)),
            Data::MatrixF32(matrix) => Data::MatrixF32(matrix.mapv(func)),
            Data::None => Data::None,
        }
    }

    pub fn apply_inplace(&mut self, func: impl Fn(&mut f32)) {
        match self {
            Data::ScalarF32(scalar) => func(scalar),
            Data::VectorF32(vector) => vector.map_inplace(func),
            Data::MatrixF32(matrix) => matrix.map_inplace(func),
            _ => {}
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
