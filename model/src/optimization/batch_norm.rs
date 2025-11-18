// builtin

// external

// internal

use crate::{
    network::config_types::{batch_norm_params::BatchNormParams, learned_params::LearnedParams},
    node::NodeRef,
};

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
}
