// builtin

// external

// internal
use crate::node::Data;

pub trait Network {
    fn feedforward(&self, input: Data);

    fn backprop(&self);
}
