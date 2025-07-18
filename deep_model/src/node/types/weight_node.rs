// builtin

// external

use crate::data::data_container::DataContainer;
// internal
use crate::data::Data;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct WeightNode<'a> {
    base: NodeBase<'a>,
    dim: (usize, usize),
    learning_rate: DataContainer,
}

impl<'a> WeightNode<'a> {
    pub fn new(input_size: usize, output_size: usize, learning_rate: f32) -> WeightNode<'a> {
        WeightNode {
            base: NodeBase::new(),
            dim: (output_size, input_size),
            learning_rate: DataContainer::Parameter(Data::ScalarF32(learning_rate)),
        }
    }
}

impl<'a> Node<'a> for WeightNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Parameter
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
        if let DataContainer::Parameter(Data::MatrixF32(matrix)) = input {
            if matrix.dim() == self.dim {
                let container = DataContainer::Parameter(Data::MatrixF32(matrix));
                self.base.set_data(container);
                return;
            }
        }
        println!("[WEIGHT] type or dimension mismatch, skipping reassignment");
    }

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {}

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
        self.base.process_gradient(&self.learning_rate);
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
