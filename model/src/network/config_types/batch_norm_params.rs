// builtin

// external
use serde::{Deserialize, Serialize};

use crate::{
    data::{data_container::DataContainer, types::FlattenedData},
    network::config_types::layer_params::LayerParams,
};

// internal

#[derive(Serialize, Deserialize)]
pub struct BatchNormParams {
    is_null: bool,
    normalization: NormParams,
    scales: LayerParams,
    shifts: LayerParams,
}

// TODO:
impl BatchNormParams {
    pub fn new(
        normalization: NormParams,
        scales: LayerParams,
        shifts: LayerParams,
    ) -> BatchNormParams {
        BatchNormParams {
            is_null: false,
            normalization,
            scales,
            shifts,
        }
    }

    pub fn null() -> BatchNormParams {
        BatchNormParams {
            is_null: true,
            normalization: NormParams::null(),
            scales: LayerParams::null(),
            shifts: LayerParams::null(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct NormParams {
    is_null: bool,
    dim: Vec<usize>,
    mean: Vec<f32>,
    variance: Vec<f32>,
    decay: f32,
}

// TODO:
impl NormParams {
    pub fn new(mean: &DataContainer, variance: &DataContainer, decay: f32) -> NormParams {
        let dim = mean.dim().1.to_vec();

        let mean_flattened: FlattenedData = mean.flatten_to_vec();
        let variance_flattened: FlattenedData = variance.flatten_to_vec();

        if let FlattenedData::Singular(mean_vec) = mean_flattened {
            if let FlattenedData::Singular(variance_vec) = variance_flattened {
                return NormParams {
                    is_null: false,
                    dim,
                    mean: mean_vec,
                    variance: variance_vec,
                    decay,
                };
            }
        }

        panic!("Invalid DataContainer format to coerce to parameters! Expected DataContainer::Parameter");
    }

    pub fn null() -> NormParams {
        NormParams {
            is_null: true,
            dim: Vec::new(),
            mean: Vec::new(),
            variance: Vec::new(),
            decay: 0.0,
        }
    }
}
