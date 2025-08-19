// builtin

// external

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// internal
use crate::data::data_container::DataContainer;
use crate::data::Data;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::optimization::momentum::MomentumType;

pub struct WeightNode<'a> {
    base: NodeBase<'a>,
    dim: (usize, usize),
    learning_rate: DataContainer,
    descent_type: MomentumType,
    decay: DataContainer,
}

impl<'a> WeightNode<'a> {
    pub fn new(input_size: usize, output_size: usize, learning_rate: f32) -> WeightNode<'a> {
        let mut base = NodeBase::new();

        let scale = f32::sqrt(6.0 / (input_size + output_size) as f32);
        let initial_weights: Array2<f32> =
            Array2::random((output_size, input_size), Uniform::new(-scale, scale));
        base.set_data(DataContainer::Parameter(Data::MatrixF32(initial_weights)));

        WeightNode {
            base,
            dim: (output_size, input_size),
            learning_rate: DataContainer::Parameter(Data::ScalarF32(learning_rate)),
            descent_type: MomentumType::None,
            decay: DataContainer::Parameter(Data::ScalarF32(0.95)),
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
        if let MomentumType::Nesterov = self.descent_type {
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

        match self.descent_type {
            MomentumType::None => self.base.process_gradient(&self.learning_rate),
            MomentumType::Base => self.base.process_momentum(&self.learning_rate, &self.decay),
            MomentumType::Nesterov => self.base.process_momentum(&self.learning_rate, &self.decay),
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
