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
pub struct MeanSquaredError;

impl MeanSquaredError {
    fn error(expected: &Data, actual: &Data) -> Data {
        data_apply_vector(
            expected,
            actual,
            MeanSquaredError::error_calc,
            |ans, pred| ans.dim() == pred.dim(),
            "MSE",
        )
    }

    fn error_calc(expected: ArrayView1<f32>, actual: ArrayView1<f32>) -> f32 {
        let mut sum: f32 = 0.0;
        let length: f32 = expected.dim() as f32;

        for (ans, pred) in expected.iter().zip(actual.iter()) {
            let val = f32::powi(ans - pred, 2);
            sum += val;
        }

        sum / (2.0 * length)
    }

    fn diff(expected: &Data, actual: &Data, wrt_expected: bool) -> Data {
        data_diff_vector(
            expected,
            actual,
            |ans: ArrayView1<f32>, pred: ArrayView1<f32>| {
                MeanSquaredError::diff_calc(ans, pred, wrt_expected)
            },
            |ans, pred| ans.dim() == pred.dim(),
            "MSE",
        )
    }

    fn diff_calc(
        expected: ArrayView1<f32>,
        actual: ArrayView1<f32>,
        wrt_expected: bool,
    ) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::with_capacity(expected.len());
        let length: f32 = expected.dim() as f32;
        for (ans, pred) in expected.iter().zip(actual.iter()) {
            if wrt_expected {
                result.push((ans - pred) / length);
            } else {
                result.push(-(ans - pred) / length);
            }
        }
        result
    }
}

impl LossType for MeanSquaredError {
    fn apply(&self, expected: &DataContainer, actual: &DataContainer) -> DataContainer {
        container_apply(expected, actual, MeanSquaredError::error, "MSE")
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
            |ans, pred| MeanSquaredError::diff(ans, pred, wrt_expected),
            "MSE",
        )
    }

    fn name(&self) -> &str {
        "mean_squared_error"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(MeanSquaredError)
    }
}
