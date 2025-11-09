// builtin

// external

// internal
use crate::data::data_container::DataContainer;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::regularization::dropout::NetworkMode;

const DELTA: f32 = 1e-6;

pub struct NormalizationNode<'a> {
    base: NodeBase<'a>,
    scale: DataContainer,
    running_mean: DataContainer,
    running_var: DataContainer,
    decay: f32,
    mode: NetworkMode,
}

impl<'a> NormalizationNode<'a> {
    pub fn new(decay: f32) -> NormalizationNode<'a> {
        NormalizationNode {
            base: NodeBase::new(),
            scale: DataContainer::Empty,
            running_mean: DataContainer::zero(),
            running_var: DataContainer::one(),
            decay,
            mode: NetworkMode::None,
        }
    }

    pub fn from_parameters(
        mean: DataContainer,
        variance: DataContainer,
        decay: f32,
    ) -> NormalizationNode<'a> {
        NormalizationNode {
            base: NodeBase::new(),
            scale: DataContainer::Empty,
            running_mean: mean,
            running_var: variance,
            decay,
            mode: NetworkMode::None,
        }
    }
}

impl<'a> Node<'a> for NormalizationNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() == 0 {
            self.base.add_input(this, input);
        } else {
            println!("[BATCH NORM] Node's maximum input capacity reached (1). Skipping assignment, consider using an extra node instead.");
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

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        if self.get_inputs().len() == 0 {
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut input_ref = inputs.get(0).unwrap().borrow_mut();
        let mut data = input_ref.get_data();

        if self.mode == NetworkMode::Train {
            let mut mean = data.average_batch();
            let mut variance: DataContainer = data.variance_batch();

            let inverse_scale = variance.apply_elementwise(|f| 1.0 / f32::sqrt(f + DELTA));
            data.minus_assign(&mean);
            data.times_assign(&inverse_scale);

            self.base.set_data(data);

            if self.running_mean.dim().1.len() != mean.dim().1.len() {
                self.running_mean = mean.apply_elementwise(|f| f * self.decay);
            } else {
                self.running_mean.apply_inplace(|f| *f *= self.decay);
                mean.apply_inplace(|f| *f *= 1.0 - self.decay);
                self.running_mean.sum_assign(&mean);
            }

            if self.running_var.dim().1.len() != variance.dim().1.len() {
                self.running_var = mean.apply_elementwise(|f| f * self.decay);
            } else {
                self.running_var.apply_inplace(|f| *f *= self.decay);
                let var_update = variance.apply_elementwise(|f| f * (1.0 - self.decay));
                self.running_var.sum_assign(&var_update);
            }

            variance.apply_inplace(|f| *f = f32::sqrt(*f + DELTA));
            self.scale = variance;
        } else if self.mode == NetworkMode::Inference {
            data.minus_assign(&self.running_mean);
            let inverse_std_dev = self
                .running_var
                .apply_elementwise(|f| 1.0 / f32::sqrt(f + DELTA));
            data.times_assign(&inverse_std_dev);

            self.base.set_data(data);
        }
    }

    fn set_data(&mut self, _data: DataContainer) {
        panic!("[BATCH NORM] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        for node in self.get_inputs() {
            let grad = self.base.get_gradient();
            let scaled = grad.times(&self.scale);

            node.borrow_mut().add_gradient(&scaled);

            if node.borrow().should_process_backprop() {
                node.borrow_mut().apply_jacobian();
            }
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_mode(&mut self, new_mode: NetworkMode) {
        self.mode = new_mode;
    }

    fn set_momentum(&mut self, _momentum: DataContainer) {
        println!("[BATCH NORM] Unsupported Operation: Cannot set momentum of an operation node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!(
            "[BATCH NORM] Unsupported Operation: Cannot set learning rate of an operation node"
        );
    }

    fn save_parameters(&self) -> LearnedParams {
        println!("[BATCH NORM] Unsupported Operation: Cannot save parameters of an operation node");
        LearnedParams::null()
    }
}
