// builtin

// external

// internal

use crate::node::{Data, Node};

pub struct WeightNode<'a> {
    outputs: Vec<&'a dyn Node<'a>>,
    weights: Data,
}

impl<'a> Node<'a> for WeightNode<'a> {
    fn add_output(&mut self, output: &'a dyn Node<'a>) {
        todo!()
    }

    fn get_inputs(&self) -> &Vec<&dyn Node<'a>> {
        todo!()
    }

    fn get_outputs(&self) -> &Vec<&dyn Node<'a>> {
        todo!()
    }

    fn get_data(&self) -> &Data {
        todo!()
    }

    fn apply_operation(&self) {
        todo!()
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
