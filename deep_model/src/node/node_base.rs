// builtin
use std::rc::Rc;

// external

// internal
use crate::node::{Data, NodeRef};

pub struct NodeBase<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
}

impl<'a> NodeBase<'a> {
    pub fn new() -> NodeBase<'a> {
        NodeBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
        }
    }
}

impl<'a> NodeBase<'a> {
    pub fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        input.borrow_mut().add_output(this);
        self.inputs.push(Rc::clone(input));
    }

    pub fn add_output(&mut self, output: &NodeRef<'a>) {
        self.outputs.push(Rc::clone(output));
    }

    pub fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    pub fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    pub fn get_data(&mut self) -> Data {
        self.data.clone()
    }

    pub fn set_data(&mut self, data: Data) {
        self.data = data;
    }
}
