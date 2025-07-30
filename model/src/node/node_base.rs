// builtin

// external

use crate::data::data_container::DataContainer;
// internal
use crate::node::NodeRef;

pub struct NodeBase<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: DataContainer,
    grad_count: usize,
    grad: DataContainer,
}

impl<'a> NodeBase<'a> {
    pub fn new() -> NodeBase<'a> {
        NodeBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: DataContainer::zero(),
            grad_count: 0,
            grad: DataContainer::zero(),
        }
    }
}

impl<'a> NodeBase<'a> {
    pub fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        input.borrow_mut().add_output(this);
        self.inputs.push(NodeRef::clone(input));
    }

    pub fn add_output(&mut self, output: &NodeRef<'a>) {
        self.outputs.push(NodeRef::clone(output));
    }

    pub fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    pub fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    pub fn get_data(&mut self) -> DataContainer {
        self.data.clone()
    }

    pub fn set_data(&mut self, data: DataContainer) {
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
        self.grad = DataContainer::zero();
    }

    pub fn add_to_gradient(&mut self, component: &DataContainer) {
        self.grad = self.grad.plus(component);
    }

    pub fn get_gradient(&self) -> &DataContainer {
        &self.grad
    }

    pub fn process_gradient(&mut self, learning_rate: &DataContainer) {
        let update = self.grad.average_batch().times(learning_rate);

        self.data = self.data.minus(&update);
    }
}
