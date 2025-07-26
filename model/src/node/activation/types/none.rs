// builtin

// external

// internal

use crate::node::activation::activation_function::ActivationType;

#[derive(Debug)]
pub struct LinearActivation;

impl ActivationType for LinearActivation {
    fn apply(&self, input: f32) -> f32 {
        input
    }

    fn diff(&self, _input: f32) -> f32 {
        1.0
    }

    fn name(&self) -> &str {
        "none"
    }

    fn copy(&self) -> Box<dyn ActivationType> {
        Box::new(LinearActivation)
    }
}
