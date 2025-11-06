// builtin

// external
use ndarray::Array1;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::learned_params::LearnedParams,
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
        weights: LearnedParams,
        biases: LearnedParams,
        activation: String,
        dropout_probability: f32,
        is_inference: bool,
    },
    Softmax {
        input_size: usize,
        output_size: usize,
        weights: LearnedParams,
        biases: LearnedParams,
        activation: String,
        dropout_probability: f32,
        is_inference: bool,
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

        UnitParams::Linear {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            is_inference: unit.borrow().is_inference(),
            dropout_probability: unit.borrow().get_mask_type().probability(),
        }
    }

    pub fn from_softmax_unit<'a>(unit: &UnitContainer<'a, SoftmaxUnit<'a>>) -> UnitParams {
        let unit_ref = unit.borrow();

        let input_size = unit_ref.get_input_size();
        let output_size = unit_ref.get_output_size();

        let weights = unit_ref.get_weights_params();
        let biases = unit_ref.get_biases_params();

        let activation = unit_ref.get_activation().to_string();

        UnitParams::Softmax {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            is_inference: unit.borrow().is_inference(),
            dropout_probability: unit.borrow().get_mask_type().probability(),
        }
    }

    pub fn new_linear(
        input_size: usize,
        output_size: usize,
        activation_function: &str,
        mask_type: UnitMaskType,
        is_inference: bool,
    ) -> UnitParams {
        let weights_dim: Vec<usize> = vec![output_size, input_size];
        let biases_dim: Vec<usize> = vec![output_size];

        let weights = UnitParams::generate_new_weights(input_size, output_size);
        let biases: Vec<f32> = vec![0.0; output_size];

        let activation = activation_function.to_string();

        UnitParams::Linear {
            input_size,
            output_size,
            weights: LearnedParams::new_from_parameters(weights_dim, weights),
            biases: LearnedParams::new_from_parameters(biases_dim, biases),
            activation,
            dropout_probability: mask_type.probability(),
            is_inference,
        }
    }

    pub fn new_softmax(
        input_size: usize,
        output_size: usize,
        activation_function: &str,
        mask_type: UnitMaskType,
        is_inference: bool,
    ) -> UnitParams {
        let weights_dim: Vec<usize> = vec![output_size, input_size];
        let biases_dim: Vec<usize> = vec![output_size];

        let weights = UnitParams::generate_new_weights(input_size, output_size);
        let biases: Vec<f32> = vec![0.0; output_size];

        let activation = activation_function.to_string();

        UnitParams::Softmax {
            input_size,
            output_size,
            weights: LearnedParams::new_from_parameters(weights_dim, weights),
            biases: LearnedParams::new_from_parameters(biases_dim, biases),
            activation,
            dropout_probability: mask_type.probability(),
            is_inference,
        }
    }

    fn generate_new_weights(input_size: usize, output_size: usize) -> Vec<f32> {
        let scale = f32::sqrt(6.0 / (input_size + output_size) as f32);
        let initial_weights: Array1<f32> =
            Array1::random(output_size * input_size, Uniform::new(-scale, scale));

        initial_weights.to_vec()
    }
}
