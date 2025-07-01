// builtin
use std::{mem::take, rc::Rc};

// external

// internal
use crate::node::{Data, Node, NodeRef};

pub struct WeightNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
    dim: (usize, usize),
}

impl<'a> WeightNode<'a> {
    pub fn new(input_size: usize, output_size: usize) -> WeightNode<'a> {
        return WeightNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
            dim: (output_size, input_size),
        };
    }
}

impl<'a> Node<'a> for WeightNode<'a> {
    fn add_input(&mut self, _this: &NodeRef<'a>, _input: &NodeRef<'a>) {}

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.outputs.push(Rc::clone(output));
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    fn set_data(&mut self, input: Data) {
        if let Data::MatrixF32(matrix) = input {
            if matrix.dim() == self.dim {
                self.data = Data::MatrixF32(matrix);
                return;
            }
        }
        println!("[WEIGHT] type or dimension mismatch, skipping reassignment");
    }

    fn get_data(&mut self) -> Data {
        take(&mut self.data)
    }

    fn apply_operation(&mut self) {}

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
