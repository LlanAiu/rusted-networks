// builtin

// external

// internal

use ndarray::{Array1, Array2, ArrayView1};

use crate::data::{data_container::DataContainer, Data};

fn warn_containers(first: &DataContainer, second: &DataContainer, operation: &str) {
    let first_variant = first.container_name();
    let second_variant = second.container_name();
    println!(
        "DataContainer::Empty returned for unsupported data container type pair for operation [{operation}]: {first_variant} and {second_variant}"
    );
}

fn warn_data(first: &Data, second: &Data, operation: &str) {
    let first_variant = first.variant_name();
    let second_variant = second.variant_name();
    println!(
        "Data::None returned for unsupported data type pair for operation [{operation}]: {first_variant} and {second_variant}"
    );
}

fn warn_dim(operation: &str) {
    println!("Data::None returned for mismatched dimensions for operation [{operation}]")
}

pub fn container_apply(
    expected: &DataContainer,
    actual: &DataContainer,
    func: impl Fn(&Data, &Data) -> Data,
    operation: &str,
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
                new_batch.push(func(ans, pred));
            }

            DataContainer::Batch(new_batch)
        }
        (DataContainer::Inference(ans), DataContainer::Inference(pred)) => {
            let new_data = func(ans, pred);

            DataContainer::Inference(new_data)
        }
        _ => {
            warn_containers(expected, actual, operation);
            DataContainer::Empty
        }
    }
}

pub fn data_apply_vector(
    expected: &Data,
    actual: &Data,
    func: impl Fn(ArrayView1<f32>, ArrayView1<f32>) -> f32,
    check: impl Fn(&ArrayView1<f32>, &ArrayView1<f32>) -> bool,
    operation: &str,
) -> Data {
    match (expected, actual) {
        (Data::VectorF32(ans), Data::VectorF32(pred)) => {
            let ans_view = ans.view();
            let pred_view = pred.view();

            if check(&ans_view, &pred_view) {
                let result = func(ans_view, pred_view);
                Data::ScalarF32(result)
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        (Data::MatrixF32(ans), Data::MatrixF32(pred)) => {
            let ans_flat = ans.flatten();
            let pred_flat = pred.flatten();

            let ans_view = ans_flat.view();
            let pred_view = pred_flat.view();

            if check(&ans_view, &pred_view) {
                let result = func(ans_view, pred_view);
                Data::ScalarF32(result)
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        _ => {
            warn_data(expected, actual, operation);
            Data::None
        }
    }
}

pub fn data_diff_vector(
    expected: &Data,
    actual: &Data,
    func: impl Fn(ArrayView1<f32>, ArrayView1<f32>) -> Vec<f32>,
    check: impl Fn(&ArrayView1<f32>, &ArrayView1<f32>) -> bool,
    operation: &str,
) -> Data {
    match (expected, actual) {
        (Data::VectorF32(ans), Data::VectorF32(pred)) => {
            let ans_view = ans.view();
            let pred_view = pred.view();

            if check(&ans_view, &pred_view) {
                let result = func(ans_view, pred_view);
                Data::VectorF32(Array1::from_vec(result))
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        (Data::MatrixF32(ans), Data::MatrixF32(pred)) => {
            let dim = ans.dim();

            let ans_flat = ans.flatten();
            let pred_flat = pred.flatten();

            let ans_view = ans_flat.view();
            let pred_view = pred_flat.view();

            if check(&ans_view, &pred_view) {
                let result = func(ans_view, pred_view);
                let matrix = Array2::from_shape_vec(dim, result)
                    .expect("Failed to coerce Jacobian into matrix format");
                Data::MatrixF32(matrix)
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        _ => {
            warn_data(expected, actual, operation);
            Data::None
        }
    }
}

pub fn data_apply_scalar(
    expected: &Data,
    actual: &Data,
    func: impl Fn(f32, f32) -> f32,
    check: impl Fn(&Array1<f32>, &Array1<f32>) -> bool,
    operation: &str,
) -> Data {
    match (expected, actual) {
        (Data::ScalarF32(ans), Data::VectorF32(pred)) => {
            let ans_wrapped: Array1<f32> = Array1::from_elem(1, *ans);

            if check(&ans_wrapped, pred) {
                let result = func(*ans, pred[0]);
                Data::ScalarF32(result)
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        (Data::VectorF32(ans), Data::VectorF32(pred)) => {
            if check(ans, pred) {
                let result = func(ans[0], pred[0]);
                Data::ScalarF32(result)
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        _ => {
            warn_data(expected, actual, operation);
            Data::None
        }
    }
}

pub fn data_diff_scalar(
    expected: &Data,
    actual: &Data,
    func: impl Fn(f32, f32) -> Vec<f32>,
    check: impl Fn(&Array1<f32>, &Array1<f32>) -> bool,
    operation: &str,
) -> Data {
    match (expected, actual) {
        (Data::VectorF32(ans), Data::VectorF32(pred)) => {
            if check(ans, pred) {
                let result = func(ans[0], pred[0]);
                Data::VectorF32(Array1::from_vec(result))
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        (Data::ScalarF32(ans), Data::VectorF32(pred)) => {
            let ans_wrapped: Array1<f32> = Array1::from_elem(1, *ans);
            if check(&ans_wrapped, pred) {
                let result = func(*ans, pred[0]);
                Data::VectorF32(Array1::from_vec(result))
            } else {
                warn_dim(operation);
                Data::None
            }
        }
        _ => {
            warn_data(expected, actual, operation);
            Data::None
        }
    }
}
