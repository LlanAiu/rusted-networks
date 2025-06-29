// builtin

// external

// internal

use crate::node::{Data, Node, NodeRef};

pub struct WeightNode<'a> {
    outputs: Vec<&'a dyn Node<'a>>,
    weights: Data,
}

impl<'a> Node<'a> for WeightNode<'a> {
    fn add_output(&mut self, output: NodeRef<'a>) {
        todo!()
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        todo!()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        todo!()
    }

    fn get_data(&self) -> &Data {
        todo!()
    }

    fn apply_operation(&mut self) {
        todo!()
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
