// builtin
use std::rc::Rc;

// external

// internal
use crate::node::{Data, NodeRef};

pub struct NodeBase<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
    grad_count: usize,
    grad: Data,
}

impl<'a> NodeBase<'a> {
    pub fn new() -> NodeBase<'a> {
        NodeBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
            grad_count: 0,
            grad: Data::None,
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

    pub fn reset_grad_count(&mut self) {
        self.grad_count = 0;
    }

    pub fn increment_grad_count(&mut self) {
        self.grad_count += 1;
    }

    pub fn should_process_backprop(&self) -> bool {
        self.grad_count == self.outputs.len()
    }

    pub fn reset_gradient(&mut self) {
        self.grad = Data::None;
    }

    pub fn add_to_gradient(&mut self, component: &Data) {
        self.grad = self.grad.sum(component);
    }

    pub fn get_gradient(&self) -> &Data {
        &self.grad
    }
}
