// builtin

// external

use crate::data::data_container::DataContainer;
// internal
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct ExpectedResponseNode<'a> {
    base: NodeBase<'a>,
    dim: &'a [usize],
}

impl<'a> ExpectedResponseNode<'a> {
    pub fn new(dim: &'a [usize]) -> ExpectedResponseNode<'a> {
        return ExpectedResponseNode {
            base: NodeBase::new(),
            dim,
        };
    }
}

impl<'a> Node<'a> for ExpectedResponseNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::ExpectedResponse
    }

    fn add_input(&mut self, _this: &NodeRef<'a>, _input: &NodeRef<'a>) {}

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.base.add_output(output);
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_outputs()
    }

    fn set_data(&mut self, input: DataContainer) {
        match input {
            DataContainer::Parameter(_) | DataContainer::Empty => {
                println!("[RESPONSE] type or dimension mismatch, skipping reassignment");
            }
            _ => {
                if input.dim().1 == self.dim {
                    self.base.set_data(input);
                }
            }
        }
    }

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {}

    fn add_gradient(&mut self, _grad: &DataContainer) {
        self.base.increment_grad_count();
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
