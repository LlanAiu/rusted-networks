// builtin

// external

// internal
use crate::data::data_container::DataContainer;
use crate::network::config_types::batch_norm_params::NormParams;
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

    fn normalize_train(&mut self, mut data: DataContainer) {
        let mean = data.average_batch();
        let mut variance: DataContainer = data.variance_batch();

        let inverse_scale = variance.apply_elementwise(|f| 1.0 / f32::sqrt(f + DELTA));
        data.minus_assign(&mean);
        data.times_assign(&inverse_scale);
        self.base.set_data(data);

        self.update_running_mean(&mean);
        self.update_running_variance(&variance);

        variance.apply_inplace(|f| *f = f32::sqrt(*f + DELTA));
        self.scale = variance;
    }

    fn update_running_mean(&mut self, mean: &DataContainer) {
        if self.running_mean.dim() != mean.dim() {
            self.running_mean = mean.apply_elementwise(|f| f * self.decay);
        } else {
            self.running_mean.apply_inplace(|f| *f *= self.decay);
            let mean_update = mean.apply_elementwise(|f| f * (1.0 - self.decay));
            self.running_mean.sum_assign(&mean_update);
        }
    }

    fn update_running_variance(&mut self, variance: &DataContainer) {
        if self.running_var.dim() != variance.dim() {
            self.running_var = variance.apply_elementwise(|f| f * self.decay);
        } else {
            self.running_var.apply_inplace(|f| *f *= self.decay);
            let var_update = variance.apply_elementwise(|f| f * (1.0 - self.decay));
            self.running_var.sum_assign(&var_update);
        }
    }

    fn normalize_inference(&mut self, mut data: DataContainer) {
        data.minus_assign(&self.running_mean);
        let inverse_std_dev = self
            .running_var
            .apply_elementwise(|f| 1.0 / f32::sqrt(f + DELTA));
        data.times_assign(&inverse_std_dev);

        self.base.set_data(data);
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
            println!("[NORMALIZE] Node's maximum input capacity reached (1). Skipping assignment, consider using an extra node instead.");
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
        let data = input_ref.get_data();

        if self.mode == NetworkMode::Train {
            self.normalize_train(data);
        } else if self.mode == NetworkMode::Inference {
            self.normalize_inference(data);
        } else {
            panic!("[NORMALIZE] Tried to run batch norm in NetworkMode::None!")
        }
    }

    fn set_data(&mut self, _data: DataContainer) {
        panic!("[NORMALIZE] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    //TODO: fix this gradient calculation
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
        println!("[NORMALIZE] Unsupported Operation: Cannot set momentum of an operation node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!(
            "[NORMALIZE] Unsupported Operation: Cannot set learning rate of an operation node"
        );
    }

    fn save_parameters(&self) -> LearnedParams {
        let params = NormParams::new(&self.running_mean, &self.running_var, self.decay);

        LearnedParams::new_batch_norm(params)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::{
        data::{data_container::DataContainer, Data},
        node::{
            types::{input_node::InputNode, normalization_node::NormalizationNode},
            NodeRef,
        },
        regularization::dropout::NetworkMode,
    };

    #[test]
    fn test_normalization() {
        let norm: NodeRef = NodeRef::new(NormalizationNode::new(0.95));

        let input: NodeRef = NodeRef::new(InputNode::new(vec![5]));

        norm.borrow_mut().add_input(&norm, &input);
        norm.borrow_mut().set_mode(NetworkMode::Train);

        let data: Vec<Data> = vec![
            Data::VectorF32(arr1(&[1.0, 3.0, 1.0, 2.0, 3.0])),
            Data::VectorF32(arr1(&[2.0, 2.0, 3.0, 1.0, 1.0])),
            Data::VectorF32(arr1(&[3.0, 1.0, 2.0, 3.0, 2.0])),
        ];

        let batch: DataContainer = DataContainer::Batch(data);
        input.borrow_mut().set_data(batch);

        norm.borrow_mut().apply_operation();

        let output: DataContainer = norm.borrow_mut().get_data();
        println!("Normalized Output {:?}", output);

        let data2: Vec<Data> = vec![
            Data::VectorF32(arr1(&[2.0, 6.0, 2.0, 4.0, 6.0])),
            Data::VectorF32(arr1(&[4.0, 4.0, 6.0, 2.0, 2.0])),
            Data::VectorF32(arr1(&[6.0, 2.0, 4.0, 6.0, 4.0])),
        ];

        let batch2: DataContainer = DataContainer::Batch(data2);
        input.borrow_mut().set_data(batch2);

        norm.borrow_mut().apply_operation();

        let output2: DataContainer = norm.borrow_mut().get_data();
        println!("Normalized Output 2 {:?}", output2);

        norm.borrow_mut().set_mode(NetworkMode::Inference);

        let inference: DataContainer =
            DataContainer::Inference(Data::VectorF32(arr1(&[1.0, 3.0, 1.0, 2.0, 3.0])));
        input.borrow_mut().set_data(inference);

        norm.borrow_mut().apply_operation();

        let inference_output: DataContainer = norm.borrow_mut().get_data();
        println!("Inference Normalized Output {:?}", inference_output);
    }
}
