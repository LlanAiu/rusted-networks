// builtin

// external
use ndarray::arr1;

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    node::loss::loss_function::LossType,
};

#[derive(Debug)]
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    fn epsilon() -> f32 {
        0.0000001
    }

    fn entropy(expected: &Data, actual: &Data) -> Data {
        match (expected, actual) {
            (Data::ScalarF32(ans), Data::VectorF32(pred)) => {
                if pred.dim() == 1 {
                    let result = BinaryCrossEntropy::entropy_calc(*ans, pred[0]);
                    Data::ScalarF32(result)
                } else {
                    BinaryCrossEntropy::warn_dim();
                    Data::None
                }
            }
            (Data::VectorF32(ans), Data::VectorF32(pred)) => {
                if ans.dim() == 1 && pred.dim() == 1 {
                    let result = BinaryCrossEntropy::entropy_calc(ans[0], pred[0]);
                    Data::ScalarF32(result)
                } else {
                    BinaryCrossEntropy::warn_dim();
                    Data::None
                }
            }
            _ => {
                BinaryCrossEntropy::warn_data(expected, actual);
                Data::None
            }
        }
    }

    fn entropy_calc(expected: f32, actual: f32) -> f32 {
        let epsilon = BinaryCrossEntropy::epsilon();
        let safe_actual = f32::clamp(actual, epsilon, 1.0 - epsilon);

        let exp_term = -expected * f32::ln(safe_actual);
        let neg_term = -(1.0 - expected) * f32::ln(1.0 - safe_actual);

        exp_term + neg_term
    }

    fn diff(expected: &Data, actual: &Data, wrt_expected: bool) -> Data {
        match (expected, actual) {
            (Data::VectorF32(ans), Data::VectorF32(pred)) => {
                if ans.dim() == 1 && pred.dim() == 1 {
                    let result = BinaryCrossEntropy::diff_calc(ans[0], pred[0], wrt_expected);
                    Data::VectorF32(arr1(&[result]))
                } else {
                    BinaryCrossEntropy::warn_dim();
                    Data::None
                }
            }
            (Data::ScalarF32(ans), Data::VectorF32(pred)) => {
                if pred.dim() == 1 {
                    let result = BinaryCrossEntropy::diff_calc(*ans, pred[0], wrt_expected);
                    Data::VectorF32(arr1(&[result]))
                } else {
                    BinaryCrossEntropy::warn_dim();
                    Data::None
                }
            }
            _ => {
                BinaryCrossEntropy::warn_data(expected, actual);
                Data::None
            }
        }
    }

    fn diff_calc(expected: f32, actual: f32, wrt_expected: bool) -> f32 {
        let epsilon = BinaryCrossEntropy::epsilon();
        let safe_actual = f32::clamp(actual, epsilon, 1.0 - epsilon);

        if wrt_expected {
            -f32::ln(safe_actual) + f32::ln(1.0 - safe_actual)
        } else {
            -expected / safe_actual + (1.0 - expected) / (1.0 - safe_actual)
        }
    }

    fn warn_containers(first: &DataContainer, second: &DataContainer) {
        let first_variant = first.container_name();
        let second_variant = second.container_name();
        println!(
            "DataContainer::Empty returned for unsupported data container type pair for operation [BINARY_ENTROPY]: {first_variant} and {second_variant}"
        )
    }

    fn warn_data(first: &Data, second: &Data) {
        let first_variant = first.variant_name();
        let second_variant = second.variant_name();
        println!(
            "Data::None returned for unsupported data type pair for operation [BINARY_ENTROPY]: {first_variant} and {second_variant}"
        )
    }

    fn warn_dim() {
        println!("Data::None returned for mismatched dimensions for operation [BINARY_ENTROPY]")
    }
}

impl LossType for BinaryCrossEntropy {
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
                    new_batch.push(BinaryCrossEntropy::entropy(ans, pred));
                }

                DataContainer::Batch(new_batch)
            }
            (DataContainer::Inference(ans), DataContainer::Inference(pred)) => {
                let new_data = BinaryCrossEntropy::entropy(ans, pred);

                DataContainer::Inference(new_data)
            }
            _ => {
                BinaryCrossEntropy::warn_containers(expected, actual);
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
                    new_batch.push(BinaryCrossEntropy::diff(ans, pred, wrt_expected));
                }

                DataContainer::Batch(new_batch)
            }
            (DataContainer::Inference(ans), DataContainer::Inference(pred)) => {
                let new_data = BinaryCrossEntropy::diff(ans, pred, wrt_expected);

                DataContainer::Inference(new_data)
            }
            _ => {
                BinaryCrossEntropy::warn_containers(expected, actual);
                DataContainer::Empty
            }
        }
    }

    fn name(&self) -> &str {
        "binary_cross_entropy"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(BinaryCrossEntropy)
    }
}
