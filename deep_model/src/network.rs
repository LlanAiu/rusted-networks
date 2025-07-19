// builtin

// external

// internal
pub mod types;
use crate::data::data_container::DataContainer;

pub trait Network {
    fn feedforward(&self, input: DataContainer);

    fn backprop(&self);
}
