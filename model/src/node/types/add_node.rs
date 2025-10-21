// builtin

// external

// internal
use crate::data::data_container::DataContainer;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct AddNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> AddNode<'a> {
    pub fn new() -> AddNode<'a> {
        AddNode {
            base: NodeBase::new(),
        }
    }
}

impl<'a> Node<'a> for AddNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        self.base.add_input(this, input);
    }

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.base.add_output(output);
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_outputs()
    }

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        if self.get_inputs().len() == 0 {
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut first_ref = inputs.get(0).unwrap().borrow_mut();
        let mut sum = first_ref.get_data();

        for i in 1..inputs.len() {
            let mut node_ref = inputs[i].borrow_mut();
            let data = node_ref.get_data();
            sum = sum.plus(&data);
        }

        self.base.set_data(sum);
    }

    fn set_data(&mut self, _data: DataContainer) {
        panic!("[ADD] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        for node in self.get_inputs() {
            node.borrow_mut().add_gradient(self.base.get_gradient());
            if node.borrow().should_process_backprop() {
                node.borrow_mut().apply_jacobian();
            }
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_momentum(&mut self, _momentum: DataContainer) {
        println!("[ADD] Unsupported Operation: Cannot set momentum of an operation node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!("[ADD] Unsupported Operation: Cannot set learning rate of an operation node");
    }

    fn save_parameters(&self) -> LearnedParams {
        println!("[ADD] Unsupported Operation: Cannot save parameters of an operation node");
        LearnedParams::null()
    }
}
