// builtin

// external
use ndarray::ArrayView1;

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    node::loss::{
        helpers::{container_apply, data_apply_vector, data_diff_vector},
        loss_function::LossType,
    },
};

#[derive(Debug)]
pub struct BaseCrossEntropy;

impl BaseCrossEntropy {
    fn epsilon() -> f32 {
        1e-7
    }

    fn error(expected: &Data, actual: &Data) -> Data {
        data_apply_vector(
            expected,
            actual,
            BaseCrossEntropy::error_calc,
            |ans, pred| ans.dim() == pred.dim(),
            "CROSS_ENTROPY",
        )
    }

    fn error_calc(expected: ArrayView1<f32>, actual: ArrayView1<f32>) -> f32 {
        let mut sum: f32 = 0.0;

        for (i, ans) in expected.iter().enumerate() {
            let epsilon = BaseCrossEntropy::epsilon();
            let mut pred = actual
                .get(i)
                .expect("[CROSS_ENTROPY] Dimension mismatch between expected and actual data");

            if *pred <= epsilon {
                pred = &epsilon;
            }

            let val = -ans * f32::ln(*pred);
            sum += val;
        }

        sum
    }

    fn diff(expected: &Data, actual: &Data, wrt_expected: bool) -> Data {
        data_diff_vector(
            expected,
            actual,
            |ans, pred| BaseCrossEntropy::diff_calc(ans, pred, wrt_expected),
            |ans, pred| ans.dim() == pred.dim(),
            "CROSS_ENTROPY",
        )
    }

    fn diff_calc(
        expected: ArrayView1<f32>,
        actual: ArrayView1<f32>,
        wrt_expected: bool,
    ) -> Vec<f32> {
        let epsilon = BaseCrossEntropy::epsilon();
        let mut result = Vec::with_capacity(expected.len());
        for (&ans, &pred) in expected.iter().zip(actual.iter()) {
            let pred_safe = if pred <= epsilon { epsilon } else { pred };
            if wrt_expected {
                result.push(-f32::ln(pred_safe));
            } else {
                result.push(-ans / pred_safe);
            }
        }
        result
    }
}

impl LossType for BaseCrossEntropy {
    fn apply(&self, expected: &DataContainer, actual: &DataContainer) -> DataContainer {
        container_apply(expected, actual, BaseCrossEntropy::error, "CROSS_ENTROPY")
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
            |ans, pred| BaseCrossEntropy::diff(ans, pred, wrt_expected),
            "CROSS_ENTROPY",
        )
    }

    fn name(&self) -> &str {
        "base_cross_entropy"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(BaseCrossEntropy)
    }
}
