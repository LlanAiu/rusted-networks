// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::learned_params::LearnedParams,
    node::{node_base::NodeBase, Node, NodeRef, NodeType},
    regularization::dropout::NetworkMode,
};

pub struct ElementSumNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> ElementSumNode<'a> {
    pub fn new() -> ElementSumNode<'a> {
        ElementSumNode {
            base: NodeBase::new(),
        }
    }
}

impl<'a> Node<'a> for ElementSumNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() == 0 {
            self.base.add_input(this, input);
        } else {
            println!("[ELEMENT_SUM] Node's maximum input capacity reached (1). Skipping assignment, consider using an extra node instead.");
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

    fn set_data(&mut self, _data: DataContainer) {
        println!("[ELEMENT_SUM] Unsupported Operation: Cannot set data of an operation node");
    }

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        let inputs = self.base.get_inputs();

        for input in inputs {
            input.borrow_mut().apply_operation();
        }

        let input_ref = inputs.get(0).unwrap();
        let input_data = input_ref.borrow_mut().get_data();

        let res = input_data.element_sum();

        self.base.set_data(res);
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
        if self.base.get_inputs().len() == 0 {
            self.base.reset_gradient();
            return;
        }

        let input = self.base.get_inputs().get(0).unwrap();
        let input_data = input.borrow_mut().get_data();

        let scale = input_data.apply_elementwise(|_f| 1.0);
        let prev_grad = self.base.get_gradient();

        let update = prev_grad.times(&scale);

        input.borrow_mut().add_gradient(&update);

        if input.borrow().should_process_backprop() {
            input.borrow_mut().apply_jacobian();
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_momentum(&mut self, _momentum: DataContainer) {
        println!("[ELEMENT_SUM] Unsupported Operation: Cannot set momentum of an operation node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!(
            "[ELEMENT_SUM] Unsupported Operation: Cannot set learning rate of an operation node"
        );
    }

    fn save_parameters(&self) -> LearnedParams {
        println!(
            "[ELEMENT_SUM] Unsupported Operation: Cannot save parameters of an operation node"
        );
        LearnedParams::null()
    }

    fn set_mode(&mut self, _new_mode: NetworkMode) {}
}
