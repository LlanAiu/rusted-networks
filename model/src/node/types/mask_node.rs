// builtin

// external

use crate::data::Data;
// internal
use crate::data::data_container::DataContainer;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::regularization::dropout::NetworkMode;

pub struct MaskNode<'a> {
    base: NodeBase<'a>,
    dim: Vec<usize>,
    mask_probability: f32,
    mode: NetworkMode,
}

impl<'a> MaskNode<'a> {
    pub fn new(dim: Vec<usize>, mask_probability: f32) -> MaskNode<'a> {
        return MaskNode {
            base: NodeBase::new(),
            dim,
            mask_probability,
            mode: NetworkMode::None,
        };
    }
}

impl<'a> Node<'a> for MaskNode<'a> {
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

    fn set_data(&mut self, _input: DataContainer) {
        panic!("[MASK] Unsupported Operation: Mask data must be generated on a random basis");
    }

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        match self.mode {
            NetworkMode::Inference => {
                let mask: DataContainer = DataContainer::Parameter(Data::one());
                self.base.set_data(mask);
            }
            NetworkMode::Train => {
                let mut data: Data = Data::bernoulli(self.mask_probability, &self.dim);
                let scale: Data = Data::ScalarF32(1.0 / self.mask_probability);
                data.times_assign(&scale);

                let mask: DataContainer = DataContainer::Parameter(data);
                self.base.set_data(mask);
            }
            NetworkMode::None => {
                panic!("Network mode is set to Mode::None, which shouldn't happen for either inference/train procedures");
            }
        }
    }

    fn add_gradient(&mut self, _grad: &DataContainer) {
        self.base.increment_grad_count();
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_mode(&mut self, new_mode: NetworkMode) {
        self.mode = new_mode;
    }

    fn set_momentum(&mut self, _momentum: DataContainer) {
        println!("[MASK] Unsupported Operation: Cannot set momentum of a mask node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!("[MASK] Unsupported Operation: Cannot set learning rate of a mask node");
    }

    fn save_parameters(&self) -> LearnedParams {
        println!("[MASK] Unsupported Operation: Cannot save parameters of a mask node");
        LearnedParams::null()
    }
}
