// builtin

// external

// internal
use crate::data::{data_container::operations::plus::ContainerPlus, Data};
pub mod operations;

#[derive(Clone)]
pub enum DataContainer {
    Batch(Vec<Data>),
    Inference(Data),
    Parameter(Data),
    Empty,
}

pub enum ContainerType {
    Batch,
    Inference,
    Parameter,
    Empty,
}

impl DataContainer {
    pub fn data_with_type(data: Data, container_type: ContainerType) -> DataContainer {
        match container_type {
            ContainerType::Inference => DataContainer::Inference(data),
            ContainerType::Parameter => DataContainer::Parameter(data),
            _ => {
                println!("Invalid container type to wrap singular data instance");
                DataContainer::Empty
            }
        }
    }

    pub fn container_type(&self) -> ContainerType {
        match self {
            DataContainer::Batch(_) => ContainerType::Batch,
            DataContainer::Inference(_) => ContainerType::Inference,
            DataContainer::Parameter(_) => ContainerType::Parameter,
            DataContainer::Empty => ContainerType::Empty,
        }
    }

    pub fn container_name(&self) -> &str {
        match self {
            DataContainer::Batch(_) => "Batch",
            DataContainer::Inference(_) => "Inference",
            DataContainer::Parameter(_) => "Parameter",
            DataContainer::Empty => "Empty",
        }
    }
}

impl DataContainer {
    fn warn_operation(this: &DataContainer, other: &DataContainer, operation: &str) {
        let this_type = this.container_name();
        let other_type = other.container_name();
        println!(
            "DataContainer::Empty returned on unsupported container type pair for operation [{operation}]: {this_type} and {other_type}!"
        );
    }

    pub fn plus(&self, other: &DataContainer) -> DataContainer {
        match (self, other) {
            (DataContainer::Batch(batch1), DataContainer::Batch(batch2)) => {
                ContainerPlus::sum_batches(batch1, batch2)
            }
            (DataContainer::Batch(batch), DataContainer::Parameter(data)) => {
                ContainerPlus::sum_batch_data(batch, data)
            }
            (DataContainer::Inference(data1), DataContainer::Inference(data2)) => {
                ContainerPlus::sum_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Inference(data1), DataContainer::Parameter(data2)) => {
                ContainerPlus::sum_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data), DataContainer::Batch(batch)) => {
                ContainerPlus::sum_batch_data(batch, data)
            }
            (DataContainer::Parameter(data1), DataContainer::Inference(data2)) => {
                ContainerPlus::sum_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data1), DataContainer::Parameter(data2)) => {
                ContainerPlus::sum_data(data1, data2, ContainerType::Parameter)
            }
            _ => {
                DataContainer::warn_operation(self, other, "PLUS");
                DataContainer::Empty
            }
        }
    }
}
