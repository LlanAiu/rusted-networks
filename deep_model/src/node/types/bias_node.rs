// builtin
use std::{mem::take, rc::Rc};

// external

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
        BiasNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
            dim,
        }
    }
}

impl<'a> Node<'a> for BiasNode<'a> {
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
        if let Data::VectorF32(vec) = input {
            if vec.dim() == self.dim {
                self.data = Data::VectorF32(vec);
                return;
            }
        }
        println!("[BIAS] type or dimension mismatch, skipping reassignment");
    }

    fn get_data(&mut self) -> Data {
        take(&mut self.data)
    }

    fn apply_operation(&mut self) {}

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
