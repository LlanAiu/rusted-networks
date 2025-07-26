// builtin

// external

// internal
pub mod types;
use crate::data::data_container::DataContainer;

pub trait Network {
    fn predict(&self, input: DataContainer) -> DataContainer;

    fn train(&self, input: DataContainer, response: DataContainer);
}
