// builtin
use std::{mem::take, rc::Rc};

// external
use ndarray::Array1;

// internal
use crate::node::{Data, Node, NodeRef};

pub struct BiasNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
    dim: usize,
}

impl<'a> BiasNode<'a> {
    pub fn new(dim: usize) -> BiasNode<'a> {
        let data: Data = Data::VectorF32(Array1::zeros(dim));

        BiasNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data,
            dim,
        }
    }

    pub fn set_data(&mut self, input: Array1<f32>) {
        if input.dim() == self.dim {
            self.data = Data::VectorF32(input);
        } else {
            println!("[BIAS] dimension mismatch, skipping reassignment");
        }
    }
}

impl<'a> Node<'a> for BiasNode<'a> {
    fn add_output(&mut self, output: NodeRef<'a>) {
        self.outputs.push(Rc::clone(&output));
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    fn get_data(&mut self) -> Data {
        take(&mut self.data)
    }

    fn apply_operation(&mut self) {}

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
