// builtin

// external

// internal
use crate::data::{
    data_container::{ContainerType, DataContainer},
    Data,
};

pub struct ContainerSquareRoot;

impl ContainerSquareRoot {
    pub fn square_root_batch(batch: &Vec<Data>) -> DataContainer {
        let mut new_data: Vec<Data> = Vec::new();
        for data in batch.iter() {
            new_data.push(data.sqrt());
        }

        DataContainer::Batch(new_data)
    }

    pub fn square_root_data(data: &Data, result_type: ContainerType) -> DataContainer {
        let new_data: Data = data.sqrt();

        DataContainer::data_with_type(new_data, result_type)
    }
}
