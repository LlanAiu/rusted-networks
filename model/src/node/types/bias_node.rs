// builtin

// external
use ndarray::Array1;

// internal
use crate::data::data_container::DataContainer;
use crate::data::Data;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::node_base::adaptive_learning_base::NodeLearningDecay;
use crate::node::node_base::momentum_base::NodeMomentum;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::optimization::learning_decay::LearningDecayType;
use crate::optimization::momentum::DescentType;
use crate::regularization::dropout::NetworkMode;

pub struct BiasNode<'a> {
    base: NodeBase<'a>,
    dim: usize,
    momentum_base: NodeMomentum,
    learning_base: NodeLearningDecay,
}

impl<'a> BiasNode<'a> {
    pub fn new(
        dim: usize,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> BiasNode<'a> {
        let mut base = NodeBase::new();

        let initial_biases: Array1<f32> = Array1::zeros(dim);
        base.set_data(DataContainer::Parameter(Data::VectorF32(initial_biases)));

        let momentum_base = NodeMomentum::new(descent_type);
        let learning_base = NodeLearningDecay::new(decay_type);

        BiasNode {
            base,
            dim,
            momentum_base,
            learning_base,
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
        let mut data = self.base.get_data();
        self.momentum_base.alter_data(&mut data);

        data
    }

    fn apply_operation(&mut self) {}

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        let mut update: DataContainer = self.base.get_gradient().average_batch();
        self.learning_base.update_learning_rate(&update);
        self.learning_base.scale_update(&mut update);

        if self.momentum_base.is_momentum_update() {
            let momentum_update: &DataContainer = self.momentum_base.get_momentum_update(&update);
            self.base.update_gradient(momentum_update);
        } else {
            self.base.update_gradient(&update);
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_momentum(&mut self, momentum: DataContainer) {
        self.momentum_base.set_momentum(momentum);
    }

    fn set_learning_rate(&mut self, learning_rate: DataContainer) {
        self.learning_base.set_learning_rate(learning_rate);
    }

    fn save_parameters(&self) -> LearnedParams {
        let data = self.base.get_data();

        if let DataContainer::Parameter(Data::VectorF32(vec)) = data {
            let parameters = vec.to_vec();

            let dim: Vec<usize> = vec![self.dim];
            let momentum = self.momentum_base.get_momentum_save();
            let learning_rate = self.learning_base.get_learning_rate_save();

            return LearnedParams::new(dim, parameters, momentum, learning_rate);
        }

        panic!("[BIAS] Unexpected data type for biases");
    }

    fn set_mode(&mut self, _new_mode: NetworkMode) {}
}
