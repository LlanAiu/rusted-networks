// builtin

// external

// internal

use crate::{
    data::data_container::DataContainer,
    network::config_types::{
        batch_norm_params::BatchNormParams, layer_params::LayerParams,
        learned_params::LearnedParams,
    },
    node::NodeRef,
};

#[derive(Clone)]
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
    decay: f32,
    normalization: NodeRef<'a>,
    scales: NodeRef<'a>,
    shifts: NodeRef<'a>,
}

impl<'a> BatchNormModule<'a> {
    pub fn new(
        decay: f32,
        normalization: &NodeRef<'a>,
        scales: &NodeRef<'a>,
        shifts: &NodeRef<'a>,
    ) -> BatchNormModule<'a> {
        BatchNormModule {
            decay,
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
