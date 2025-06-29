// builtin

// external

// internal

use std::rc::Rc;

use ndarray::Array1;

use crate::node::{Data, Node, NodeRef};

pub struct InputNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
}

impl<'a> InputNode<'a> {
    pub fn new(dim: usize) -> InputNode<'a> {
        let data: Data = Data::VectorF32(Array1::zeros(dim));

        return InputNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data,
        };
    }

    pub fn set_data(&mut self, input: Array1<f32>) {
        self.data = Data::VectorF32(input);
    }
}

impl<'a> Node<'a> for InputNode<'a> {
    fn add_output(&mut self, output: NodeRef<'a>) {
        self.outputs.push(Rc::clone(&output));
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn apply_operation(&mut self) {}

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
