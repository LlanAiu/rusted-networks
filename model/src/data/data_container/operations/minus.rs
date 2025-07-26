// builtin

// external

// internal

use crate::data::{
    data_container::{ContainerType, DataContainer},
    Data,
};

pub struct ContainerMinus;

impl ContainerMinus {
    fn warn_batch(batch1: &Vec<Data>, batch2: &Vec<Data>) {
        println!(
            "DataContainer::Empty returned on batch size mismatch for operation [MINUS]: {} and {}",
            batch1.len(),
            batch2.len()
        );
    }

    pub fn subtract_batches(batch1: &Vec<Data>, batch2: &Vec<Data>) -> DataContainer {
        if batch1.len() != batch2.len() {
            ContainerMinus::warn_batch(batch1, batch2);
            return DataContainer::Empty;
        }

        let mut new_data: Vec<Data> = Vec::new();
        for (first, second) in batch1.iter().zip(batch2.iter()) {
            new_data.push(first.minus(second));
        }

        DataContainer::Batch(new_data)
    }

    pub fn subtract_batch_data(batch: &Vec<Data>, data: &Data) -> DataContainer {
        let mut new_data: Vec<Data> = Vec::new();
        for batch_data in batch.iter() {
            new_data.push(batch_data.minus(data));
        }

        DataContainer::Batch(new_data)
    }

    pub fn subtract_data_batch(data: &Data, batch: &Vec<Data>) -> DataContainer {
        let mut new_data: Vec<Data> = Vec::new();
        for batch_data in batch.iter() {
            new_data.push(data.minus(batch_data));
        }

        DataContainer::Batch(new_data)
    }

    pub fn subtract_data(data1: &Data, data2: &Data, result_type: ContainerType) -> DataContainer {
        let new_data: Data = data1.minus(data2);

        DataContainer::data_with_type(new_data, result_type)
    }
}
