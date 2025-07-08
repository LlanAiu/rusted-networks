// builtin

// external

// internal

use crate::node::activation::activation_function::ActivationType;

#[derive(Debug)]
pub struct ReLUActivation;

impl ActivationType for ReLUActivation {
    fn apply(&self, input: f32) -> f32 {
        if input < 0.0 {
            return 0.0;
        }
        input
    }

    fn diff(&self, input: f32) -> f32 {
        if input < 0.0 {
            return 0.0;
        }
        1.0
    }

    fn name(&self) -> &str {
        "relu"
    }

    fn copy(&self) -> Box<dyn ActivationType> {
        Box::new(ReLUActivation)
    }
}
