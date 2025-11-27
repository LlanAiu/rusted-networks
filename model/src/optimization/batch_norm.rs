// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::{
        batch_norm_params::BatchNormParams, layer_params::LayerParams,
        learned_params::LearnedParams,
    },
    node::NodeRef,
};

#[derive(Serialize, Deserialize, Clone)]
pub enum NormalizationType {
    BatchNorm { decay: f32 },
    None,
}

impl NormalizationType {
    pub fn none() -> NormalizationType {
        NormalizationType::None
    }

    pub fn batch_norm(decay: f32) -> NormalizationType {
        NormalizationType::BatchNorm { decay }
    }
}

pub struct BatchNormModule<'a> {
    normalization: NodeRef<'a>,
    scales: NodeRef<'a>,
    shifts: NodeRef<'a>,
}

impl<'a> BatchNormModule<'a> {
    pub fn new(
        normalization: &NodeRef<'a>,
        scales: &NodeRef<'a>,
        shifts: &NodeRef<'a>,
    ) -> BatchNormModule<'a> {
        BatchNormModule {
            normalization: NodeRef::clone(normalization),
            scales: NodeRef::clone(scales),
            shifts: NodeRef::clone(shifts),
        }
    }

    pub fn get_params(&self) -> BatchNormParams {
        let norm_params = self.normalization.borrow().save_parameters();

        let scale_params = self.scales.borrow().save_parameters();

        let shift_params = self.shifts.borrow().save_parameters();

        if let LearnedParams::BatchNorm {
            params: normalization,
        } = norm_params
        {
            if let LearnedParams::Layer { params: scales } = scale_params {
                if let LearnedParams::Layer { params: shifts } = shift_params {
                    return BatchNormParams::new(normalization, scales, shifts);
                }
            }
        }

        println!("Invalid LearnedParams type from saved parameters");
        BatchNormParams::null()
    }

    pub fn set_parameters(&self, params: &BatchNormParams) {
        let scales = params.get_scales();
        let shifts = params.get_shifts();

        Self::set_node_parameters(&self.scales, scales);
        Self::set_node_parameters(&self.shifts, shifts);
    }

    fn set_node_parameters(node: &NodeRef<'a>, params: &LayerParams) {
        let parameters = params.get_parameters();
        let momentum = params.get_momentum();
        let learning_rate = params.get_learning_rate();

        node.borrow_mut().set_data(parameters);
        if !matches!(&momentum, DataContainer::Empty) {
            node.borrow_mut().set_momentum(momentum);
        }
        if !matches!(&learning_rate, DataContainer::Empty) {
            node.borrow_mut().set_learning_rate(learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::data::data_container::DataContainer;
    use crate::data::Data;
    use crate::node::types::add_node::AddNode;
    use crate::node::types::expected_response_node::ExpectedResponseNode;
    use crate::node::types::input_node::InputNode;
    use crate::node::types::loss_node::LossNode;
    use crate::node::types::multiply_node::MultiplyNode;
    use crate::node::{
        types::{
            bias_node::BiasNode, normalization_node::NormalizationNode, weight_node::WeightNode,
        },
        NodeRef,
    };
    use crate::optimization::learning_decay::LearningDecayType;
    use crate::optimization::momentum::DescentType;
    use crate::regularization::dropout::NetworkMode;

    #[test]
    fn batch_norm_module_test() {
        let norm_ref: NodeRef = NodeRef::new(NormalizationNode::new(0.95));
        let scale_ref: NodeRef = NodeRef::new(WeightNode::new_vec(
            5,
            LearningDecayType::constant(0.4),
            DescentType::none(),
        ));
        let shift_ref: NodeRef = NodeRef::new(BiasNode::new(
            5,
            LearningDecayType::constant(0.4),
            DescentType::none(),
        ));

        let add_ref: NodeRef = NodeRef::new(AddNode::new());
        let multiply_ref: NodeRef = NodeRef::new(MultiplyNode::new());
        let input_ref: NodeRef = NodeRef::new(InputNode::new(vec![5]));

        let loss_ref: NodeRef = NodeRef::new(LossNode::new("mean_squared_error"));
        let response_ref: NodeRef = NodeRef::new(ExpectedResponseNode::new(vec![5]));

        norm_ref.borrow_mut().add_input(&norm_ref, &input_ref);

        multiply_ref
            .borrow_mut()
            .add_input(&multiply_ref, &norm_ref);
        multiply_ref
            .borrow_mut()
            .add_input(&multiply_ref, &scale_ref);

        add_ref.borrow_mut().add_input(&add_ref, &multiply_ref);
        add_ref.borrow_mut().add_input(&add_ref, &shift_ref);

        loss_ref.borrow_mut().add_input(&loss_ref, &response_ref);
        loss_ref.borrow_mut().add_input(&loss_ref, &add_ref);

        norm_ref.borrow_mut().set_mode(NetworkMode::Train);

        let data: Vec<Data> = vec![
            Data::VectorF32(arr1(&[1.0, 3.0, 1.0, 2.0, 3.0])),
            Data::VectorF32(arr1(&[2.0, 2.0, 3.0, 1.0, 1.0])),
            Data::VectorF32(arr1(&[3.0, 1.0, 2.0, 3.0, 2.0])),
        ];

        let response: Vec<Data> = vec![
            Data::VectorF32(arr1(&[4.0, 8.0, 4.0, 6.0, 8.0])),
            Data::VectorF32(arr1(&[6.0, 6.0, 8.0, 4.0, 4.0])),
            Data::VectorF32(arr1(&[8.0, 4.0, 6.0, 8.0, 6.0])),
        ];

        for _i in 1..100 {
            let batch_input: DataContainer = DataContainer::Batch(data.clone());
            input_ref.borrow_mut().set_data(batch_input);
            let batch_response: DataContainer = DataContainer::Batch(response.clone());
            response_ref.borrow_mut().set_data(batch_response);

            loss_ref.borrow_mut().apply_operation();

            loss_ref.borrow_mut().add_gradient(&DataContainer::one());
            loss_ref.borrow_mut().apply_jacobian();
        }

        norm_ref.borrow_mut().set_mode(NetworkMode::Inference);
        let inference: DataContainer =
            DataContainer::Inference(Data::VectorF32(arr1(&[1.0, 3.0, 1.0, 2.0, 3.0])));
        input_ref.borrow_mut().set_data(inference);
        add_ref.borrow_mut().apply_operation();

        let output: DataContainer = add_ref.borrow_mut().get_data();
        println!("Output {:?}", output);

        let scales: DataContainer = scale_ref.borrow_mut().get_data();
        println!("Scales {:?}", scales);

        let shifts: DataContainer = shift_ref.borrow_mut().get_data();
        println!("Shifts {:?}", shifts);
    }
}
