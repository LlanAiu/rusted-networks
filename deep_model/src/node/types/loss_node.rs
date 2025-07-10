// builtin

// external

// internal
use crate::data::Data;
use crate::node::loss::loss_function::LossFunction;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct LossNode<'a> {
    base: NodeBase<'a>,
    function: LossFunction,
}

impl<'a> LossNode<'a> {
    pub fn new(function_name: &str) -> LossNode<'a> {
        LossNode {
            base: NodeBase::new(),
            function: LossFunction::new(function_name),
        }
    }
}

impl<'a> Node<'a> for LossNode<'a> {
    //INPUTS MUST BE ADDED IN THE ORDER OF: PREDICTED, ACTUAL
    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() < 2 {
            self.base.add_input(this, input);
        } else {
            println!("[LOSS] Node's maximum input capacity (2) reached. Skipping assignment.");
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
        if self.get_inputs().len() != 2 {
            println!("[ACTIVATION] Tried to apply operation on no inputs");
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

        let data = self.function.apply(second_data, first_data);

        self.base.set_data(data);
    }

    fn set_data(&mut self, _data: Data) {
        panic!("[LOSS] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &Data) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        todo!()
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
