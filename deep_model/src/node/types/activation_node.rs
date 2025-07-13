// builtin

// external

// internal
use crate::data::Data;
use crate::node::NodeType;
use crate::node::{
    activation::activation_function::ActivationFunction, node_base::NodeBase, Node, NodeRef,
};

pub struct ActivationNode<'a> {
    base: NodeBase<'a>,
    function: ActivationFunction,
}

impl<'a> ActivationNode<'a> {
    pub fn new(function_name: &str) -> ActivationNode<'a> {
        ActivationNode {
            base: NodeBase::new(),
            function: ActivationFunction::new(function_name),
        }
    }
}

impl<'a> Node<'a> for ActivationNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() == 0 {
            self.base.add_input(this, input);
        } else {
            println!("[ACTIVATION] Node's maximum input capacity reached (1). Skipping assignment, consider using an extra node instead.");
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

    fn get_data(&mut self) -> Data {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        if self.get_inputs().len() == 0 {
            println!("[ACTIVATION] Tried to apply operation on no inputs");
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut input_ref = inputs.get(0).unwrap().borrow_mut();

        let mut data = input_ref.get_data();
        self.function.apply_all(&mut data);

        self.base.set_data(data);
    }

    fn set_data(&mut self, _data: Data) {
        panic!("[ACTIVATION] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &Data) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        for node in self.get_inputs() {
            let mut grad = node.borrow_mut().get_data();
            self.function.diff_all(&mut grad);
            node.borrow_mut()
                .add_gradient(&grad.times(self.base.get_gradient()));
            if node.borrow().should_process_backprop() {
                node.borrow_mut().apply_jacobian();
            }
        }
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
