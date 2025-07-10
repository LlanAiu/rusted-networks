// builtin

// external

// internal

use crate::{data::Data, node::loss::loss_function::LossType};

#[derive(Debug)]
pub struct BaseCrossEntropy;

impl LossType for BaseCrossEntropy {
    fn apply(&self, expected: Data, actual: Data) -> Data {
        todo!()
    }

    fn diff(&self, expected: Data, actual: Data) -> Data {
        todo!()
    }

    fn name(&self) -> &str {
        "base_cross_entropy"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(BaseCrossEntropy)
    }
}
