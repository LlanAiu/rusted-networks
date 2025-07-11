// builtin

// external
use ndarray::{Array1, Array2, ArrayView1};

// internal
use crate::{data::Data, node::loss::loss_function::LossType};

#[derive(Debug)]
pub struct BaseCrossEntropy;

impl BaseCrossEntropy {
    fn epsilon() -> f32 {
        0.0000001
    }

    fn entropy(expected: ArrayView1<f32>, actual: ArrayView1<f32>) -> f32 {
        let mut sum: f32 = 0.0;

        for (i, ans) in expected.iter().enumerate() {
            let epsilon = BaseCrossEntropy::epsilon();
            let mut pred = actual
                .get(i)
                .expect("[CROSS_ENTROPY] Dimension mismatch between expected and actual data");

            if *pred <= 0.0 {
                pred = &epsilon;
            }

            let val = ans * f32::ln(*pred);
            sum -= val;
        }

        sum
    }

    fn diff(expected: ArrayView1<f32>, actual: ArrayView1<f32>, wrt_expected: bool) -> Vec<f32> {
        let epsilon = BaseCrossEntropy::epsilon();
        let mut result = Vec::with_capacity(expected.len());
        for (&ans, &pred) in expected.iter().zip(actual.iter()) {
            let pred_safe = if pred <= 0.0 { epsilon } else { pred };
            if wrt_expected {
                result.push(-f32::ln(pred_safe));
            } else {
                result.push(-ans / pred_safe);
            }
        }
        result
    }

    fn warn(first: &Data, second: &Data) {
        let first_variant = first.variant_name();
        let second_variant = second.variant_name();
        println!(
            "Data::None return for unsupported data type pair for operation [CROSS_ENTROPY]: {first_variant} and {second_variant}"
        )
    }
}

impl LossType for BaseCrossEntropy {
    fn apply(&self, expected: &Data, actual: &Data) -> Data {
        match (expected, actual) {
            (Data::MatrixF32(ans), Data::MatrixF32(pred)) => {
                let ans_flat = ans.flatten();
                let pred_flat = pred.flatten();

                Data::ScalarF32(BaseCrossEntropy::entropy(ans_flat.view(), pred_flat.view()))
            }
            (Data::VectorF32(ans), Data::VectorF32(pred)) => {
                let ans_view = ans.view();
                let pred_view = pred.view();

                Data::ScalarF32(BaseCrossEntropy::entropy(ans_view, pred_view))
            }
            _ => {
                BaseCrossEntropy::warn(expected, actual);
                Data::None
            }
        }
    }

    fn diff(&self, expected: &Data, actual: &Data, wrt_expected: bool) -> Data {
        match (expected, actual) {
            (Data::VectorF32(ans), Data::VectorF32(pred)) => {
                let ans_view = ans.view();
                let pred_view = pred.view();

                let result = BaseCrossEntropy::diff(ans_view, pred_view, wrt_expected);
                Data::VectorF32(Array1::from_vec(result))
            }
            (Data::MatrixF32(ans), Data::MatrixF32(pred)) => {
                let dim = ans.dim();

                let ans_flat = ans.flatten();
                let pred_flat = pred.flatten();

                let result =
                    BaseCrossEntropy::diff(ans_flat.view(), pred_flat.view(), wrt_expected);

                let matrix = Array2::from_shape_vec(dim, result)
                    .expect("[CROSS_ENTROPY] Failed to coerce derivative back to matrix format");

                Data::MatrixF32(matrix)
            }
            _ => {
                BaseCrossEntropy::warn(expected, actual);
                Data::None
            }
        }
    }

    fn name(&self) -> &str {
        "base_cross_entropy"
    }

    fn copy(&self) -> Box<dyn LossType> {
        Box::new(BaseCrossEntropy)
    }
}
