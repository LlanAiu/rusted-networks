// builtin
use std::fmt::Debug;

// external

// internal
use crate::{
    data::Data,
    node::loss::registry::{init_loss_registry, LossRegistry},
};

pub trait LossType: Send + Sync + Debug {
    fn apply(&self, expected: &Data, actual: &Data) -> Data;

    fn diff(&self, expected: &Data, actual: &Data, wrt_expected: bool) -> Data;

    fn name(&self) -> &str;

    fn copy(&self) -> Box<dyn LossType>;
}

pub struct LossFunction {
    loss_type: Box<dyn LossType>,
}

impl LossFunction {
    pub fn new(loss_name: &str) -> LossFunction {
        init_loss_registry();

        LossFunction {
            loss_type: LossRegistry::get(loss_name),
        }
    }

    pub fn apply(&self, expected: &Data, actual: &Data) -> Data {
        self.loss_type.apply(expected, actual)
    }

    pub fn get_jacobian(&self, expected: &Data, actual: &Data, wrt_expected: bool) -> Data {
        self.loss_type.diff(expected, actual, wrt_expected)
    }
}
