// builtin

// external
use serde::{Deserialize, Serialize};

use crate::network::config_types::{batch_norm_params::NormParams, layer_params::LayerParams};

// internal

#[derive(Serialize, Deserialize)]
pub enum LearnedParams {
    Layer { params: LayerParams },
    BatchNorm { params: NormParams },
    None,
}

impl LearnedParams {
    pub fn null() -> LearnedParams {
        LearnedParams::None
    }

    pub fn new_layer(params: LayerParams) -> LearnedParams {
        LearnedParams::Layer { params }
    }

    pub fn new_batch_norm(params: NormParams) -> LearnedParams {
        LearnedParams::BatchNorm { params }
    }
}
