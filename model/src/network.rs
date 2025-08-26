// builtin

// external

// internal
use crate::{data::data_container::DataContainer, network::config_types::Config};
pub mod config_types;
pub mod types;

pub trait Network {
    fn reset(&mut self);

    fn predict(&self, input: DataContainer) -> DataContainer;

    fn train(&self, input: DataContainer, response: DataContainer);

    fn create_config(&self) -> Config;
}
