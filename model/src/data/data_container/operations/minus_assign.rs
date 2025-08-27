// builtin

// external

// internal
use crate::data::Data;

pub struct ContainerMinusAssign;

impl ContainerMinusAssign {
    fn warn_batch(batch1: &Vec<Data>, batch2: &Vec<Data>) {
        println!(
            "Mutation failed on batch size mismatch for operation [PLUS_INPLACE]: {} and {}",
            batch1.len(),
            batch2.len()
        );
    }

    pub fn minus_batches(batch1: &mut Vec<Data>, batch2: &Vec<Data>) {
        if batch1.len() != batch2.len() {
            Self::warn_batch(batch1, batch2);
            return;
        }

        for (first, second) in batch1.into_iter().zip(batch2.iter()) {
            first.minus_assign(second);
        }
    }

    pub fn minus_batch_data(batch: &mut Vec<Data>, data: &Data) {
        for batch_data in batch.into_iter() {
            batch_data.minus_assign(data);
        }
    }

    pub fn minus_data(l_data: &mut Data, r_data: &Data) {
        l_data.minus_assign(r_data);
    }
}
