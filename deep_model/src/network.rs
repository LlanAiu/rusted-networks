// builtin

// external

// internal
use crate::data::Data;

pub trait Network {
    fn feedforward(&self, input: Data);

    fn backprop(&self);
}
