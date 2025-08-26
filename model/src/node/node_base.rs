// builtin

// external

// internal
use crate::data::data_container::DataContainer;
use crate::node::NodeRef;

pub struct NodeBase<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: DataContainer,
    grad_count: usize,
    grad: DataContainer,
    momentum: DataContainer,
    is_grad_null: bool,
    is_momentum_null: bool,
}

impl<'a> NodeBase<'a> {
    pub fn new() -> NodeBase<'a> {
        NodeBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: DataContainer::zero(),
            grad_count: 0,
            grad: DataContainer::zero(),
            momentum: DataContainer::zero(),
            is_grad_null: true,
            is_momentum_null: true,
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

    pub fn get_data(&self) -> DataContainer {
        self.data.clone()
    }

    pub fn get_nesterov_data(&self) -> DataContainer {
        self.data.plus(&self.momentum)
    }

    pub fn set_data(&mut self, data: DataContainer) {
        self.data = data;
    }

    pub fn reset_momentum(&mut self) {
        self.momentum = DataContainer::zero();
        self.is_momentum_null = true;
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
        self.is_grad_null = true;
    }

    pub fn add_to_gradient(&mut self, component: &DataContainer) {
        if self.is_grad_null {
            self.grad = self.grad.plus(component);
            self.is_grad_null = false;
        } else {
            self.grad.sum_assign(component);
        }
    }

    pub fn get_gradient(&self) -> &DataContainer {
        &self.grad
    }

    pub fn process_gradient(&mut self, learning_rate: &DataContainer) {
        let mut update = self.grad.average_batch();
        update.times_assign(learning_rate);

        self.data.minus_assign(&update);
    }

    pub fn process_momentum(&mut self, learning_rate: &DataContainer, decay: &DataContainer) {
        let mut update = self.grad.average_batch();
        update.times_assign(learning_rate);

        if self.is_momentum_null {
            self.momentum = DataContainer::neg_one().times(&update);
        } else {
            self.momentum.times_assign(decay);
            self.momentum.minus_assign(&update);
        }

        self.data.sum_assign(&self.momentum);
    }
}
