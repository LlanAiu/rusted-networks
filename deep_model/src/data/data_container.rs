// builtin

// external

// internal
use crate::data::{
    data_container::operations::{
        matmul::ContainerMatMul, minus::ContainerMinus, plus::ContainerPlus, times::ContainerTimes,
    },
    Data,
};
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

    pub fn minus(&self, other: &DataContainer) -> DataContainer {
        match (self, other) {
            (DataContainer::Batch(batch1), DataContainer::Batch(batch2)) => {
                ContainerMinus::subtract_batches(batch1, batch2)
            }
            (DataContainer::Batch(batch), DataContainer::Parameter(data)) => {
                ContainerMinus::subtract_batch_data(batch, data)
            }
            (DataContainer::Inference(data1), DataContainer::Inference(data2)) => {
                ContainerMinus::subtract_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Inference(data1), DataContainer::Parameter(data2)) => {
                ContainerMinus::subtract_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data), DataContainer::Batch(batch)) => {
                ContainerMinus::subtract_data_batch(data, batch)
            }
            (DataContainer::Parameter(data1), DataContainer::Inference(data2)) => {
                ContainerMinus::subtract_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data1), DataContainer::Parameter(data2)) => {
                ContainerMinus::subtract_data(data1, data2, ContainerType::Parameter)
            }
            _ => {
                DataContainer::warn_operation(self, other, "MINUS");
                DataContainer::Empty
            }
        }
    }

    pub fn times(&self, other: &DataContainer) -> DataContainer {
        match (self, other) {
            (DataContainer::Batch(batch1), DataContainer::Batch(batch2)) => {
                ContainerTimes::multiply_batches(batch1, batch2)
            }
            (DataContainer::Batch(batch), DataContainer::Parameter(data)) => {
                ContainerTimes::multiply_batch_data(batch, data)
            }
            (DataContainer::Inference(data1), DataContainer::Inference(data2)) => {
                ContainerTimes::multiply_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Inference(data1), DataContainer::Parameter(data2)) => {
                ContainerTimes::multiply_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data), DataContainer::Batch(batch)) => {
                ContainerTimes::multiply_batch_data(batch, data)
            }
            (DataContainer::Parameter(data1), DataContainer::Inference(data2)) => {
                ContainerTimes::multiply_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data1), DataContainer::Parameter(data2)) => {
                ContainerTimes::multiply_data(data1, data2, ContainerType::Parameter)
            }
            _ => {
                DataContainer::warn_operation(self, other, "PLUS");
                DataContainer::Empty
            }
        }
    }

    pub fn matmul(&self, other: &DataContainer) -> DataContainer {
        match (self, other) {
            (DataContainer::Batch(batch1), DataContainer::Batch(batch2)) => {
                ContainerMatMul::matmul_batches(batch1, batch2)
            }
            (DataContainer::Batch(batch), DataContainer::Parameter(data)) => {
                ContainerMatMul::matmul_batch_data(batch, data)
            }
            (DataContainer::Inference(data1), DataContainer::Inference(data2)) => {
                ContainerMatMul::matmul_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Inference(data1), DataContainer::Parameter(data2)) => {
                ContainerMatMul::matmul_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data), DataContainer::Batch(batch)) => {
                ContainerMatMul::matmul_data_batch(data, batch)
            }
            (DataContainer::Parameter(data1), DataContainer::Inference(data2)) => {
                ContainerMatMul::matmul_data(data1, data2, ContainerType::Inference)
            }
            (DataContainer::Parameter(data1), DataContainer::Parameter(data2)) => {
                ContainerMatMul::matmul_data(data1, data2, ContainerType::Parameter)
            }
            _ => {
                DataContainer::warn_operation(self, other, "MINUS");
                DataContainer::Empty
            }
        }
    }

    pub fn apply_function_ref(&self, func: impl Fn(&Data) -> Data) -> DataContainer {
        match self {
            DataContainer::Batch(batch) => {
                let mapped: Vec<Data> = batch.iter().map(|f| func(f)).collect();
                DataContainer::Batch(mapped)
            }
            DataContainer::Inference(data) => DataContainer::Inference(func(data)),
            DataContainer::Parameter(data) => DataContainer::Parameter(func(data)),
            _ => DataContainer::Empty,
        }
    }

    pub fn apply_function(self, func: impl Fn(Data) -> Data) -> DataContainer {
        match self {
            DataContainer::Batch(batch) => {
                let mapped: Vec<Data> = batch.iter().map(|f| func(f.to_owned())).collect();
                DataContainer::Batch(mapped)
            }
            DataContainer::Inference(data) => DataContainer::Inference(func(data)),
            DataContainer::Parameter(data) => DataContainer::Parameter(func(data)),
            _ => DataContainer::Empty,
        }
    }
}
