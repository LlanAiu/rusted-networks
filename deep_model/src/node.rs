// builtin
use std::{cell::RefCell, rc::Rc};

// external
use ndarray::{Array1, Array2};

// internal
pub mod types;

#[derive(Debug)]
pub enum Data {
    VectorF32(Array1<f32>),
    MatrixF32(Array2<f32>),
    None,
}

impl Data {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Data::VectorF32(_) => "VectorF32",
            Data::MatrixF32(_) => "MatrixF32",
            Data::None => "None",
        }
    }
}

impl Default for Data {
    fn default() -> Self {
        Data::None
    }
}

pub type NodeRef<'a> = Rc<RefCell<dyn Node<'a> + 'a>>;

pub trait Node<'a> {
    fn add_output(&mut self, output: NodeRef<'a>);

    fn get_inputs(&self) -> &Vec<NodeRef<'a>>;

    fn get_outputs(&self) -> &Vec<NodeRef<'a>>;

    fn get_data(&mut self) -> Data;

    fn apply_operation(&mut self);

    fn get_jacobian(&self) -> Data;
}
