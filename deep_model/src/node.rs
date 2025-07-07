// builtin
use std::{cell::RefCell, rc::Rc};

// external
use ndarray::{Array1, Array2};

// internal
pub mod activation;
pub mod node_base;
pub mod types;

#[derive(Debug, Clone)]
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

impl Data {
    pub fn sum(&self, other: &Data) -> Data {
        match self {
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this + vec);
                    }
                }
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this + matrix);
                    }
                }
                Data::None
            }
            Data::None => other.clone(),
        }
    }

    pub fn minus(&self, other: &Data) -> Data {
        match self {
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this - vec);
                    }
                }
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this - matrix);
                    }
                }
                Data::None
            }
            Data::None => other.clone().scale(-1.0),
        }
    }

    pub fn times(&self, other: &Data) -> Data {
        match self {
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this * vec);
                    }
                }
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this * matrix);
                    }
                }
                Data::None
            }
            Data::None => Data::None,
        }
    }

    pub fn scale(&self, scalar: f32) -> Data {
        match self {
            Data::VectorF32(this) => Data::VectorF32(this * scalar),
            Data::MatrixF32(this) => Data::MatrixF32(this * scalar),
            Data::None => Data::None,
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
    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>);

    fn add_output(&mut self, output: &NodeRef<'a>);

    fn get_inputs(&self) -> &Vec<NodeRef<'a>>;

    fn get_outputs(&self) -> &Vec<NodeRef<'a>>;

    fn set_data(&mut self, data: Data);

    fn get_data(&mut self) -> Data;

    fn apply_operation(&mut self);

    fn add_gradient(&mut self, grad: &Data);

    fn apply_jacobian(&mut self);

    fn should_process_backprop(&self) -> bool;
}
