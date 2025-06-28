// builtin

// external

use ndarray::{Array1, Array2};

// internal
pub mod types;

#[derive(Debug)]
pub enum Data {
    VectorF32(Array1<f32>),
    MatrixF32(Array2<f32>),
}

impl Data {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Data::VectorF32(_) => "VectorF32",
            Data::MatrixF32(_) => "MatrixF32",
        }
    }
}

pub trait Node<'a> {
    fn add_output(&mut self, output: &'a dyn Node<'a>);

    fn get_inputs(&self) -> &Vec<&dyn Node<'a>>;

    fn get_outputs(&self) -> &Vec<&dyn Node<'a>>;

    fn get_data(&self) -> &Data;

    fn apply_operation(&self);

    fn get_jacobian(&self) -> Data;
}
