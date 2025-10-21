// builtin

// external

// internal
use crate::data::data_container::DataContainer;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct MatrixMultiplyNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> MatrixMultiplyNode<'a> {
    pub fn new() -> MatrixMultiplyNode<'a> {
        MatrixMultiplyNode {
            base: NodeBase::new(),
        }
    }
}

impl<'a> Node<'a> for MatrixMultiplyNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() < 2 {
            self.base.add_input(this, input);
        } else {
            println!("[MATMUL] Node's maximum input capacity (2) reached. Skipping assignment, consider using an extra node instead.");
        }
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
        if self.get_inputs().len() != 2 {
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut first_ref = inputs.get(0).unwrap().borrow_mut();
        let first_data = first_ref.get_data();

        let mut second_ref = inputs.get(1).unwrap().borrow_mut();
        let second_data = second_ref.get_data();

        let res: DataContainer = first_data.matmul(&second_data);

        self.base.set_data(res);
    }

    fn set_data(&mut self, _data: DataContainer) {
        println!("[MATMUL] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        let mut first_ref = inputs.get(0).unwrap().borrow_mut();
        let mut second_ref = inputs.get(1).unwrap().borrow_mut();

        let first_data = first_ref.get_data();
        let second_data = second_ref.get_data();
        let grad = self.base.get_gradient();

        let first_grad = grad.matmul(&second_data.transpose());
        let second_grad = first_data.transpose().matmul(grad);

        first_ref.add_gradient(&first_grad);
        second_ref.add_gradient(&second_grad);

        if first_ref.should_process_backprop() {
            first_ref.apply_jacobian();
        }

        if second_ref.should_process_backprop() {
            second_ref.apply_jacobian();
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_momentum(&mut self, _momentum: DataContainer) {
        println!("[MATMUL] Unsupported Operation: Cannot set momentum of an operation node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!("[MATMUL] Unsupported Operation: Cannot set learning rate of an operation node");
    }

    fn save_parameters(&self) -> LearnedParams {
        println!("[MATMUL] Unsupported Operation: Cannot save parameters of an operation node");
        LearnedParams::null()
    }
}
