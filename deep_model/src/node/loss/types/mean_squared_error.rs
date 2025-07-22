// builtin

// external
use ndarray::{Array1, Array2, ArrayView1};

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    node::loss::loss_function::LossType,
};

#[derive(Debug)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    fn error(expected: &Data, actual: &Data) -> Data {
        match (expected, actual) {
            (Data::VectorF32(ans), Data::VectorF32(pred)) => {
                let ans_view = ans.view();
                let pred_view = pred.view();

                let result = MeanSquaredError::error_calc(ans_view, pred_view);
                Data::ScalarF32(result)
            }
            (Data::MatrixF32(ans), Data::MatrixF32(pred)) => {
                let ans_flat = ans.flatten();
                let pred_flat = pred.flatten();

                let result = MeanSquaredError::error_calc(ans_flat.view(), pred_flat.view());
                Data::ScalarF32(result)
            }
            _ => {
                MeanSquaredError::warn_data(expected, actual);
                Data::None
            }
        }
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
        match (expected, actual) {
            (Data::VectorF32(ans), Data::VectorF32(pred)) => {
                let ans_view = ans.view();
                let pred_view = pred.view();

                let result = MeanSquaredError::diff_calc(ans_view, pred_view, wrt_expected);
                Data::VectorF32(Array1::from_vec(result))
            }
            (Data::MatrixF32(ans), Data::MatrixF32(pred)) => {
                let dim = ans.dim();

                let ans_flat = ans.flatten();
                let pred_flat = pred.flatten();

                let result =
                    MeanSquaredError::diff_calc(ans_flat.view(), pred_flat.view(), wrt_expected);

                let matrix = Array2::from_shape_vec(dim, result)
                    .expect("[SQUARED_ERROR] Failed to coerce Jacobian into matrix format");

                Data::MatrixF32(matrix)
            }
            _ => {
                MeanSquaredError::warn_data(expected, actual);
                Data::None
            }
        }
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

    fn warn_containers(first: &DataContainer, second: &DataContainer) {
        let first_variant = first.container_name();
        let second_variant = second.container_name();
        println!(
            "DataContainer::Empty returned for unsupported data container type pair for operation [SQUARED_ERROR]: {first_variant} and {second_variant}"
        )
    }

    fn warn_data(first: &Data, second: &Data) {
        let first_variant = first.variant_name();
        let second_variant = second.variant_name();
        println!(
            "Data::None returned for unsupported data type pair for operation [SQUARED_ERROR]: {first_variant} and {second_variant}"
        )
    }
}

impl LossType for MeanSquaredError {
    fn apply(&self, expected: &DataContainer, actual: &DataContainer) -> DataContainer {
        match (expected, actual) {
            (DataContainer::Batch(ans_batch), DataContainer::Batch(pred_batch)) => {
                if ans_batch.len() != pred_batch.len() {
                    println!(
                        "DataContainer::Empty returned due to mismatch in batch sizes: {} and {}",
                        ans_batch.len(),
                        pred_batch.len()
                    );
                    return DataContainer::Empty;
                }

                let mut new_batch: Vec<Data> = Vec::new();
                for (ans, pred) in ans_batch.iter().zip(pred_batch.iter()) {
                    new_batch.push(MeanSquaredError::error(ans, pred));
                }

                DataContainer::Batch(new_batch)
            }
            (DataContainer::Inference(ans), DataContainer::Inference(pred)) => {
                let new_data = MeanSquaredError::error(ans, pred);

                DataContainer::Inference(new_data)
            }
            _ => {
                MeanSquaredError::warn_containers(expected, actual);
                DataContainer::Empty
            }
        }
    }

    fn diff(
        &self,
        expected: &DataContainer,
        actual: &DataContainer,
        wrt_expected: bool,
    ) -> DataContainer {
        match (expected, actual) {
            (DataContainer::Batch(ans_batch), DataContainer::Batch(pred_batch)) => {
                if ans_batch.len() != pred_batch.len() {
                    println!(
                        "DataContainer::Empty returned due to mismatch in batch sizes: {} and {}",
                        ans_batch.len(),
                        pred_batch.len()
                    );
                    return DataContainer::Empty;
                }

                let mut new_batch: Vec<Data> = Vec::new();
                for (ans, pred) in ans_batch.iter().zip(pred_batch.iter()) {
                    new_batch.push(MeanSquaredError::diff(ans, pred, wrt_expected));
                }

                DataContainer::Batch(new_batch)
            }
            (DataContainer::Inference(ans), DataContainer::Inference(pred)) => {
                let new_data = MeanSquaredError::diff(ans, pred, wrt_expected);

                DataContainer::Inference(new_data)
            }
            _ => {
                MeanSquaredError::warn_containers(expected, actual);
                DataContainer::Empty
            }
        }
    }

    fn name(&self) -> &str {
        "mean_squared_error"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(MeanSquaredError)
    }
}
