// builtin

// external
use ndarray::Array1;

// internal
use crate::data::data_container::DataContainer;
use crate::data::Data;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::optimization::momentum::DescentType;

pub struct BiasNode<'a> {
    base: NodeBase<'a>,
    dim: usize,
    learning_rate: DataContainer,
    descent_type: DescentType,
}

impl<'a> BiasNode<'a> {
    pub fn new(dim: usize, learning_rate: f32) -> BiasNode<'a> {
        let mut base = NodeBase::new();

        let initial_biases: Array1<f32> = Array1::zeros(dim);
        base.set_data(DataContainer::Parameter(Data::VectorF32(initial_biases)));

        BiasNode {
            base,
            dim,
            learning_rate: DataContainer::Parameter(Data::ScalarF32(learning_rate)),
            descent_type: DescentType::Base,
        }
    }
}

impl<'a> Node<'a> for BiasNode<'a> {
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
        if let DataContainer::Parameter(Data::VectorF32(vec)) = input {
            if vec.dim() == self.dim {
                let container = DataContainer::Parameter(Data::VectorF32(vec));
                self.base.set_data(container);
                return;
            }
        }
        println!("[BIAS] type or dimension mismatch, skipping reassignment");
    }

    fn get_data(&mut self) -> DataContainer {
        if let DescentType::Nesterov { .. } = &self.descent_type {
            return self.base.get_nesterov_data();
        }
        self.base.get_data()
    }

    fn apply_operation(&mut self) {}

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        match &self.descent_type {
            DescentType::Base => self.base.process_gradient(&self.learning_rate),
            DescentType::Momentum { decay } => {
                self.base.process_momentum(&self.learning_rate, decay)
            }
            DescentType::Nesterov { decay } => {
                self.base.process_momentum(&self.learning_rate, decay)
            }
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
