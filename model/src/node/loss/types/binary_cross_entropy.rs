// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    node::loss::{
        helpers::{container_apply, data_apply_scalar, data_diff_scalar},
        loss_function::LossType,
    },
};

#[derive(Debug)]
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    fn epsilon() -> f32 {
        0.0000001
    }

    fn error(expected: &Data, actual: &Data) -> Data {
        data_apply_scalar(
            expected,
            actual,
            BinaryCrossEntropy::error_calc,
            |ans, pred| ans.dim() == 1 && pred.dim() == 1,
            "BINARY_ENTROPY",
        )
    }

    fn error_calc(expected: f32, actual: f32) -> f32 {
        let epsilon: f32 = BinaryCrossEntropy::epsilon();
        let safe_actual: f32 = f32::clamp(actual, epsilon, 1.0 - epsilon);

        let exp_term: f32 = -expected * f32::ln(safe_actual);
        let neg_term: f32 = -(1.0 - expected) * f32::ln(1.0 - safe_actual);

        exp_term + neg_term
    }

    fn diff(expected: &Data, actual: &Data, wrt_expected: bool) -> Data {
        data_diff_scalar(
            expected,
            actual,
            |ans, pred| BinaryCrossEntropy::diff_calc(ans, pred, wrt_expected),
            |ans, pred| ans.dim() == 1 && pred.dim() == 1,
            "BINARY_ENTROPY",
        )
    }

    fn diff_calc(expected: f32, actual: f32, wrt_expected: bool) -> Vec<f32> {
        let epsilon = BinaryCrossEntropy::epsilon();
        let safe_actual = f32::clamp(actual, epsilon, 1.0 - epsilon);

        let result: f32 = if wrt_expected {
            -f32::ln(safe_actual) + f32::ln(1.0 - safe_actual)
        } else {
            -expected / safe_actual + (1.0 - expected) / (1.0 - safe_actual)
        };

        vec![result]
    }
}

impl LossType for BinaryCrossEntropy {
    fn apply(&self, expected: &DataContainer, actual: &DataContainer) -> DataContainer {
        container_apply(
            expected,
            actual,
            BinaryCrossEntropy::error,
            "BINARY_ENTROPY",
        )
    }

    fn diff(
        &self,
        expected: &DataContainer,
        actual: &DataContainer,
        wrt_expected: bool,
    ) -> DataContainer {
        container_apply(
            expected,
            actual,
            |ans: &Data, pred: &Data| BinaryCrossEntropy::diff(ans, pred, wrt_expected),
            "BINARY_ENTROPY",
        )
    }

    fn name(&self) -> &str {
        "binary_cross_entropy"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(BinaryCrossEntropy)
    }
}
