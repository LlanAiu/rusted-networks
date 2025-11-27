// builtin

// external
use ndarray::Array1;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::{batch_norm_params::BatchNormParams, layer_params::LayerParams},
    optimization::batch_norm::NormalizationType,
    regularization::dropout::UnitMaskType,
    unit::{
        types::{linear_unit::LinearUnit, softmax_unit::SoftmaxUnit},
        UnitContainer,
    },
};

#[derive(Serialize, Deserialize)]
#[serde(tag = "unit_type")]
pub enum UnitParams {
    Linear {
        input_size: usize,
        output_size: usize,
        weights: LayerParams,
        biases: LayerParams,
        activation: String,
        keep_probability: f32,
        is_inference: bool,
        norm_params: BatchNormParams,
    },
    Softmax {
        input_size: usize,
        output_size: usize,
        weights: LayerParams,
        biases: LayerParams,
        activation: String,
        keep_probability: f32,
        is_inference: bool,
        norm_params: BatchNormParams,
    },
}

impl UnitParams {
    pub fn get_weights(&self) -> DataContainer {
        match self {
            UnitParams::Linear { weights, .. } => weights.get_parameters(),
            UnitParams::Softmax { weights, .. } => weights.get_parameters(),
        }
    }

    pub fn get_biases(&self) -> DataContainer {
        match self {
            UnitParams::Linear { biases, .. } => biases.get_parameters(),
            UnitParams::Softmax { biases, .. } => biases.get_parameters(),
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            UnitParams::Linear { .. } => "UnitParam::Linear",
            UnitParams::Softmax { .. } => "UnitParam::Softmax",
        }
    }

    pub fn from_linear_unit<'a>(unit: &UnitContainer<'a, LinearUnit<'a>>) -> UnitParams {
        let unit_ref = unit.borrow();

        let input_size = unit_ref.get_input_size();
        let output_size = unit_ref.get_output_size();

        let weights = unit_ref.get_weights_params();
        let biases = unit_ref.get_biases_params();

        let activation = unit_ref.get_activation().to_string();

        let norm_params = unit_ref.get_batch_norm_params();

        UnitParams::Linear {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            is_inference: unit.borrow().is_inference(),
            keep_probability: unit.borrow().get_mask_type().probability(),
            norm_params,
        }
    }

    pub fn from_softmax_unit<'a>(unit: &UnitContainer<'a, SoftmaxUnit<'a>>) -> UnitParams {
        let unit_ref = unit.borrow();

        let input_size = unit_ref.get_input_size();
        let output_size = unit_ref.get_output_size();

        let weights = unit_ref.get_weights_params();
        let biases = unit_ref.get_biases_params();

        let activation = unit_ref.get_activation().to_string();

        let norm_params = unit_ref.get_batch_norm_params();

        UnitParams::Softmax {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            is_inference: unit.borrow().is_inference(),
            keep_probability: unit.borrow().get_mask_type().probability(),
            norm_params,
        }
    }

    pub fn new_linear(
        input_size: usize,
        output_size: usize,
        activation_function: &str,
        mask_type: UnitMaskType,
        normalization_type: NormalizationType,
        is_inference: bool,
    ) -> UnitParams {
        let weights_dim: Vec<usize> = vec![output_size, input_size];
        let biases_dim: Vec<usize> = vec![output_size];

        let weights = UnitParams::generate_new_weights(input_size, output_size);
        let biases: Vec<f32> = vec![0.0; output_size];

        let activation = activation_function.to_string();

        let norm_params: BatchNormParams = UnitParams::generate_new_norm_params(
            normalization_type,
            vec![output_size],
            output_size,
        );

        UnitParams::Linear {
            input_size,
            output_size,
            weights: LayerParams::new_from_parameters(weights_dim, weights),
            biases: LayerParams::new_from_parameters(biases_dim, biases),
            activation,
            keep_probability: mask_type.probability(),
            is_inference,
            norm_params,
        }
    }

    pub fn new_softmax(
        input_size: usize,
        output_size: usize,
        activation_function: &str,
        mask_type: UnitMaskType,
        normalization_type: NormalizationType,
        is_inference: bool,
    ) -> UnitParams {
        let weights_dim: Vec<usize> = vec![output_size, input_size];
        let biases_dim: Vec<usize> = vec![output_size];

        let weights: Vec<f32> = UnitParams::generate_new_weights(input_size, output_size);
        let biases: Vec<f32> = vec![0.0; output_size];

        let activation: String = activation_function.to_string();

        let norm_params: BatchNormParams = UnitParams::generate_new_norm_params(
            normalization_type,
            vec![output_size],
            output_size,
        );

        UnitParams::Softmax {
            input_size,
            output_size,
            weights: LayerParams::new_from_parameters(weights_dim, weights),
            biases: LayerParams::new_from_parameters(biases_dim, biases),
            activation,
            keep_probability: mask_type.probability(),
            norm_params,
            is_inference,
        }
    }

    fn generate_new_weights(input_size: usize, output_size: usize) -> Vec<f32> {
        let scale = f32::sqrt(6.0 / (input_size + output_size) as f32);
        let initial_weights: Array1<f32> =
            Array1::random(output_size * input_size, Uniform::new(-scale, scale));

        initial_weights.to_vec()
    }

    fn generate_new_norm_params(
        normalization_type: NormalizationType,
        dim: Vec<usize>,
        size: usize,
    ) -> BatchNormParams {
        match normalization_type {
            NormalizationType::BatchNorm { decay } => {
                let scales: Vec<f32> = vec![1.0; size];
                let shifts: Vec<f32> = vec![0.0; size];

                BatchNormParams::new_from_parameters(dim, decay, scales, shifts)
            }
            NormalizationType::None => BatchNormParams::null(),
        }
    }
}
