// builtin

// external

// internal
use crate::data::{
    data_container::{ContainerType, DataContainer},
    Data,
};

pub struct ContainerElementSum;

impl ContainerElementSum {
    pub fn element_sum_batch(batch: &Vec<Data>) -> DataContainer {
        let mut new_data: Vec<Data> = Vec::new();
        for data in batch.iter() {
            new_data.push(data.element_sum());
        }

        DataContainer::Batch(new_data)
    }

    pub fn element_sum_data(data: &Data, result_type: ContainerType) -> DataContainer {
        let new_data: Data = data.element_sum();

        DataContainer::data_with_type(new_data, result_type)
    }
}
