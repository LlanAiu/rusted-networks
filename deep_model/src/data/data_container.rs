// builtin

// external

// internal
pub mod operations;
use crate::data::Data;

#[derive(Clone)]
pub enum DataContainer {
    Batch(Vec<Data>),
    Inference(Data),
    Parameter(Data),
    Empty,
}

impl DataContainer {
    pub fn sum(&self, other: &DataContainer) -> DataContainer {
        match (self, other) {
            (DataContainer::Batch(this), DataContainer::Batch(other)) => {
                if this.len() != other.len() {
                    return DataContainer::Empty;
                }
                let mut new_data: Vec<Data> = Vec::new();
                for (this_data, other_data) in this.iter().zip(other.iter()) {
                    new_data.push(this_data.plus(other_data));
                }
                DataContainer::Batch(new_data)
            }
            (DataContainer::Batch(this), DataContainer::Inference(other)) => todo!(),
            (DataContainer::Batch(this), DataContainer::Parameter(other)) => todo!(),
            (DataContainer::Inference(this), DataContainer::Batch(other)) => todo!(),
            (DataContainer::Inference(this), DataContainer::Inference(other)) => todo!(),
            (DataContainer::Inference(this), DataContainer::Parameter(other)) => todo!(),
            (DataContainer::Parameter(this), DataContainer::Batch(other)) => todo!(),
            (DataContainer::Parameter(this), DataContainer::Inference(other)) => todo!(),
            (DataContainer::Parameter(this), DataContainer::Parameter(other)) => todo!(),
            _ => DataContainer::Empty,
        }
    }
}
