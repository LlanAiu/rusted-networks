// builtin

// external

use std::fmt::Debug;

use crate::node::{
    activation::registry::{init_registry, Registry},
    Data,
};

// internal

pub trait ActivationType: Send + Sync + Debug {
    fn apply(&self, input: f32) -> f32;

    fn diff(&self, input: f32) -> f32;

    fn name(&self) -> &str;

    fn copy(&self) -> Box<dyn ActivationType>;
}

pub struct ActivationFunction {
    activation_type: Box<dyn ActivationType>,
}

impl ActivationFunction {
    pub fn new(activation_type: &str) -> ActivationFunction {
        init_registry();

        ActivationFunction {
            activation_type: Registry::get(activation_type),
        }
    }

    pub fn apply_all(&self, data: &mut Data) {
        match data {
            Data::VectorF32(vec) => {
                vec.mapv_inplace(|f| self.activation_type.apply(f));
            }
            Data::MatrixF32(matrix) => {
                matrix.mapv_inplace(|f| self.activation_type.apply(f));
            }
            _ => {}
        }
    }

    pub fn diff_all(&self, data: &Data) -> Data {
        todo!()
    }
}
