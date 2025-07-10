// builtin

// external

// internal
use crate::data::Data;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct OutputNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> OutputNode<'a> {
    pub fn new() -> OutputNode<'a> {
        return OutputNode {
            base: NodeBase::new(),
        };
    }
}

impl<'a> Node<'a> for OutputNode<'a> {
    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        self.base.add_input(this, input)
    }

    fn add_output(&mut self, _output: &NodeRef<'a>) {}

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_outputs()
    }

    fn set_data(&mut self, _input: Data) {
        println!("[OUTPUT] Unsupported Operation: Cannot set data of an output node");
    }

    fn get_data(&mut self) -> Data {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        if self.base.get_inputs().len() == 0 {
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut input_ref = inputs.get(0).unwrap().borrow_mut();
        let input_data = input_ref.get_data();

        self.base.set_data(input_data);
    }

    fn add_gradient(&mut self, _grad: &Data) {}

    fn apply_jacobian(&mut self) {}

    fn should_process_backprop(&self) -> bool {
        false
    }
}
