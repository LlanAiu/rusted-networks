// builtin
use std::{mem::take, rc::Rc};

// external
use ndarray::Array2;

// internal
use crate::node::{Data, Node, NodeRef};

pub struct WeightNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
    dim: (usize, usize),
}

impl<'a> WeightNode<'a> {
    pub fn new(dim: (usize, usize)) -> WeightNode<'a> {
        let weights: Data = Data::MatrixF32(Array2::ones(dim));

        return WeightNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: weights,
            dim,
        };
    }

    pub fn set_data(&mut self, input: Array2<f32>) {
        if input.dim() == self.dim {
            self.data = Data::MatrixF32(input);
        } else {
            println!("[WEIGHT] dimension mismatch, skipping reassignment");
        }
    }
}

impl<'a> Node<'a> for WeightNode<'a> {
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
