// builtin

// external

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// internal
use crate::data::data_container::DataContainer;
use crate::data::Data;
use crate::network::config_types::layer_params::LayerParams;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::node_base::adaptive_learning_base::NodeLearningDecay;
use crate::node::node_base::momentum_base::NodeMomentum;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::optimization::learning_decay::LearningDecayType;
use crate::optimization::momentum::DescentType;
use crate::regularization::dropout::NetworkMode;

pub struct WeightNode<'a> {
    base: NodeBase<'a>,
    dim: Vec<usize>,
    momentum_base: NodeMomentum,
    learning_base: NodeLearningDecay,
}

impl<'a> WeightNode<'a> {
    pub fn new_matrix(
        input_size: usize,
        output_size: usize,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> WeightNode<'a> {
        let mut base = NodeBase::new();

        let initial_weights: DataContainer =
            Self::get_initial_weights_matrix(input_size, output_size);
        base.set_data(initial_weights);

        let momentum_base = NodeMomentum::new(descent_type);
        let learning_base = NodeLearningDecay::new(decay_type);

        WeightNode {
            base,
            dim: vec![output_size, input_size],
            momentum_base,
            learning_base,
        }
    }

    fn get_initial_weights_matrix(input_size: usize, output_size: usize) -> DataContainer {
        let scale = f32::sqrt(6.0 / (input_size + output_size) as f32);
        let initial_weights: Array2<f32> =
            Array2::random((output_size, input_size), Uniform::new(-scale, scale));
        DataContainer::Parameter(Data::MatrixF32(initial_weights))
    }

    pub fn new_vec(
        size: usize,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> WeightNode<'a> {
        let mut base = NodeBase::new();

        let initial_weights: DataContainer = Self::get_initial_weights_vec(size);
        base.set_data(initial_weights);

        let momentum_base = NodeMomentum::new(descent_type);
        let learning_base = NodeLearningDecay::new(decay_type);

        WeightNode {
            base,
            dim: vec![size],
            momentum_base,
            learning_base,
        }
    }

    fn get_initial_weights_vec(size: usize) -> DataContainer {
        let scale = f32::sqrt(6.0 / size as f32);
        let initial_weights: Array1<f32> = Array1::random(size, Uniform::new(-scale, scale));
        DataContainer::Parameter(Data::VectorF32(initial_weights))
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
        if let DataContainer::Parameter(data) = input {
            match data {
                Data::VectorF32(vec) => {
                    let check: bool = self.dim.len() == 1 && vec.dim() == self.dim[0];
                    if check {
                        let container = DataContainer::Parameter(Data::VectorF32(vec));
                        self.base.set_data(container);
                        return;
                    }
                }
                Data::MatrixF32(matrix) => {
                    let check: bool = self.dim.len() == 2
                        && matrix.dim().0 == self.dim[0]
                        && matrix.dim().1 == self.dim[1];
                    if check {
                        let container = DataContainer::Parameter(Data::MatrixF32(matrix));
                        self.base.set_data(container);
                        return;
                    }
                }
                _ => {
                    println!("[WEIGHT] type or dimension mismatch, skipping reassignment");
                }
            }
        }
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
        let container = self.base.get_data();

        if let DataContainer::Parameter(data) = container {
            let dim: Vec<usize> = self.dim.clone();
            let momentum = self.momentum_base.get_momentum_save();
            let learning_rate = self.learning_base.get_learning_rate_save();

            let parameters: Vec<f32> = match data {
                Data::VectorF32(vec) => vec.to_vec(),
                Data::MatrixF32(matrix) => matrix.flatten().to_vec(),
                _ => panic!("[WEIGHT] Unexpected data type for weights!"),
            };

            let params = LayerParams::new(dim, parameters, momentum, learning_rate);
            return LearnedParams::new_layer(params);
        }

        panic!("[WEIGHT] Unexpected data type for weights!");
    }

    fn set_mode(&mut self, _new_mode: NetworkMode) {}
}
