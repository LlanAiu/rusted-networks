// builtin

// external

// internal
use crate::data::{
    data_container::operations::{
        element_sum::ContainerElementSum, matmul::ContainerMatMul, minus::ContainerMinus,
        plus::ContainerPlus, sqrt::ContainerSquareRoot, sum_assign::ContainerSumAssign,
        times::ContainerTimes, transpose::ContainerTranspose,
    },
    Data,
};
pub mod operations;

#[derive(Clone, Debug)]
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

    pub fn container_name(&self) -> &'static str {
        match self {
            DataContainer::Batch(_) => "Batch",
            DataContainer::Inference(_) => "Inference",
            DataContainer::Parameter(_) => "Parameter",
            DataContainer::Empty => "Empty",
        }
    }

    pub fn zero() -> DataContainer {
        DataContainer::Parameter(Data::zero())
    }

    pub fn one() -> DataContainer {
        DataContainer::Parameter(Data::one())
    }

    pub fn neg_one() -> DataContainer {
        DataContainer::Parameter(Data::neg_one())
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

    fn warn_mutate(this_type: &str, other: &DataContainer, operation: &str) {
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

    pub fn sum_assign(&mut self, other: &DataContainer) {
        let variant = self.container_name();
        match (self, other) {
            (DataContainer::Batch(l_batch), DataContainer::Batch(r_batch)) => {
                ContainerSumAssign::sum_batches(l_batch, r_batch);
            }
            (DataContainer::Batch(batch), DataContainer::Inference(data)) => {
                ContainerSumAssign::sum_batch_data(batch, data);
            }
            (DataContainer::Batch(batch), DataContainer::Parameter(data)) => {
                ContainerSumAssign::sum_batch_data(batch, data);
            }
            (DataContainer::Inference(l_data), DataContainer::Inference(r_data)) => {
                ContainerSumAssign::sum_data(l_data, r_data);
            }
            (DataContainer::Inference(l_data), DataContainer::Parameter(r_data)) => {
                ContainerSumAssign::sum_data(l_data, r_data);
            }
            (DataContainer::Parameter(l_data), DataContainer::Inference(r_data)) => {
                ContainerSumAssign::sum_data(l_data, r_data);
            }
            (DataContainer::Parameter(l_data), DataContainer::Parameter(r_data)) => {
                ContainerSumAssign::sum_data(l_data, r_data);
            }
            _ => {
                DataContainer::warn_mutate(variant, other, "SUM_INPLACE");
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

    pub fn transpose(&self) -> DataContainer {
        match self {
            DataContainer::Batch(batch) => ContainerTranspose::transpose_batch(batch),
            DataContainer::Inference(data) => {
                ContainerTranspose::transpose_data(data, ContainerType::Inference)
            }
            DataContainer::Parameter(data) => {
                ContainerTranspose::transpose_data(data, ContainerType::Parameter)
            }
            DataContainer::Empty => DataContainer::Empty,
        }
    }

    pub fn element_sum(&self) -> DataContainer {
        match self {
            DataContainer::Batch(batch) => ContainerElementSum::element_sum_batch(batch),
            DataContainer::Inference(data) => {
                ContainerElementSum::element_sum_data(data, ContainerType::Inference)
            }
            DataContainer::Parameter(data) => {
                ContainerElementSum::element_sum_data(data, ContainerType::Parameter)
            }
            DataContainer::Empty => DataContainer::Empty,
        }
    }

    pub fn sqrt(&self) -> DataContainer {
        match self {
            DataContainer::Batch(batch) => ContainerSquareRoot::square_root_batch(batch),
            DataContainer::Inference(data) => {
                ContainerSquareRoot::square_root_data(data, ContainerType::Inference)
            }
            DataContainer::Parameter(data) => {
                ContainerSquareRoot::square_root_data(data, ContainerType::Parameter)
            }
            DataContainer::Empty => DataContainer::Empty,
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

    pub fn apply_elementwise<F>(&self, func: F) -> DataContainer
    where
        F: Fn(f32) -> f32 + Copy,
    {
        match self {
            DataContainer::Batch(datas) => {
                let mapped: Vec<Data> = datas
                    .iter()
                    .map(|data| data.apply_elementwise(func))
                    .collect();
                DataContainer::Batch(mapped)
            }
            DataContainer::Inference(data) => {
                let new_data = data.apply_elementwise(func);
                DataContainer::Inference(new_data)
            }
            DataContainer::Parameter(data) => {
                let new_data = data.apply_elementwise(func);
                DataContainer::Parameter(new_data)
            }
            DataContainer::Empty => todo!(),
        }
    }

    pub fn average_batch(&self) -> DataContainer {
        match self {
            DataContainer::Batch(batch) => {
                if batch.len() == 0 {
                    return DataContainer::Empty;
                }
                let len_scale: f32 = 1.0 / batch.len() as f32;
                let mut average: Data = Data::zero();
                for data in batch {
                    average = average.plus(&data);
                }
                average = average.times(&Data::ScalarF32(len_scale));
                DataContainer::Parameter(average)
            }
            _ => self.clone(),
        }
    }

    pub fn dim(&self) -> (usize, &[usize]) {
        match self {
            DataContainer::Batch(batch) => {
                if batch.len() == 0 {
                    return (0, &[]);
                }
                return (batch.len(), batch[0].dim());
            }
            DataContainer::Inference(data) => (1, data.dim()),
            DataContainer::Parameter(data) => (1, data.dim()),
            DataContainer::Empty => (0, &[]),
        }
    }
}
