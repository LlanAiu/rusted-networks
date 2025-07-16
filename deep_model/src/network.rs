// builtin

// external

// internal
pub mod types;
use crate::data::Data;

pub trait Network {
    fn feedforward(&self, input: Data);

    fn backprop(&self);
}
