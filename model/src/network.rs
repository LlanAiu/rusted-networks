// builtin

// external

// internal
use crate::data::data_container::DataContainer;
pub mod config_types;
pub mod types;

pub trait Network {
    fn predict(&self, input: DataContainer) -> DataContainer;

    fn train(&self, input: DataContainer, response: DataContainer);
}
