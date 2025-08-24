// builtin

// external

// internal

use crate::data::Data;

pub struct ContainerSumAssign;

impl ContainerSumAssign {
    fn warn_batch(batch1: &Vec<Data>, batch2: &Vec<Data>) {
        println!(
            "Mutation failed on batch size mismatch for operation [MINUS_INPLACE]: {} and {}",
            batch1.len(),
            batch2.len()
        );
    }

    pub fn sum_batches(batch1: &mut Vec<Data>, batch2: &Vec<Data>) {
        if batch1.len() != batch2.len() {
            Self::warn_batch(batch1, batch2);
            return;
        }

        for (first, second) in batch1.into_iter().zip(batch2.iter()) {
            first.sum_assign(second);
        }
    }

    pub fn sum_batch_data(batch: &mut Vec<Data>, data: &Data) {
        for batch_data in batch.into_iter() {
            batch_data.sum_assign(data);
        }
    }

    pub fn sum_data(l_data: &mut Data, r_data: &Data) {
        l_data.sum_assign(r_data);
    }
}
