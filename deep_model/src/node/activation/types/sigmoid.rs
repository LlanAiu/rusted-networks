// builtin

// external

// internal
use crate::node::activation::activation_function::ActivationType;

#[derive(Debug)]
pub struct SigmoidActivation;

impl ActivationType for SigmoidActivation {
    fn apply(&self, input: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-input))
    }

    fn diff(&self, input: f32) -> f32 {
        let sigmoid: f32 = self.apply(input);

        sigmoid * (1.0 - sigmoid)
    }

    fn name(&self) -> &str {
        "sigmoid"
    }

    fn copy(&self) -> Box<dyn ActivationType> {
        Box::new(SigmoidActivation)
    }
}
