// builtin

// external
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

// internal
use crate::{
    data::{data_container::DataContainer, Data},
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
        weights_dim: (usize, usize),
        weights: Vec<f32>,
        biases: Vec<f32>,
        activation: String,
    },
    Softmax {
        input_size: usize,
        output_size: usize,
        weights_dim: (usize, usize),
        weights: Vec<f32>,
        biases: Vec<f32>,
        activation: String,
    },
}

impl UnitParams {
    pub fn get_weights(&self) -> DataContainer {
        match self {
            UnitParams::Linear {
                weights_dim,
                weights,
                ..
            } => {
                let matrix = Array2::from_shape_vec(*weights_dim, weights.clone()).unwrap();
                DataContainer::Parameter(Data::MatrixF32(matrix))
            }
            UnitParams::Softmax {
                weights_dim,
                weights,
                ..
            } => {
                let matrix = Array2::from_shape_vec(*weights_dim, weights.clone()).unwrap();
                DataContainer::Parameter(Data::MatrixF32(matrix))
            }
        }
    }

    pub fn get_biases(&self) -> DataContainer {
        match self {
            UnitParams::Linear { biases, .. } => {
                let vec = Array1::from_vec(biases.clone());
                DataContainer::Parameter(Data::VectorF32(vec))
            }
            UnitParams::Softmax { biases, .. } => {
                let vec = Array1::from_vec(biases.clone());
                DataContainer::Parameter(Data::VectorF32(vec))
            }
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

        let weights_dim = (output_size, input_size);

        let weights = unit_ref.get_weights();
        let biases = unit_ref.get_biases();

        let activation = unit_ref.get_activation().to_string();

        UnitParams::Linear {
            input_size,
            output_size,
            weights_dim,
            weights,
            biases,
            activation,
        }
    }

    pub fn from_softmax_unit<'a>(unit: &UnitContainer<'a, SoftmaxUnit<'a>>) -> UnitParams {
        let unit_ref = unit.borrow();

        let input_size = unit_ref.get_input_size();
        let output_size = unit_ref.get_output_size();

        let weights_dim = (output_size, input_size);

        let weights = unit_ref.get_weights();
        let biases = unit_ref.get_biases();

        let activation = unit_ref.get_activation().to_string();

        UnitParams::Softmax {
            input_size,
            output_size,
            weights_dim,
            weights,
            biases,
            activation,
        }
    }

    pub fn new_linear(
        input_size: usize,
        output_size: usize,
        activation_function: &str,
    ) -> UnitParams {
        let weights_dim = (output_size, input_size);

        let weights = UnitParams::generate_new_weights(input_size, output_size);
        let biases = vec![0.0; output_size];

        let activation = activation_function.to_string();

        UnitParams::Linear {
            input_size,
            output_size,
            weights_dim,
            weights,
            biases,
            activation,
        }
    }

    pub fn new_softmax(
        input_size: usize,
        output_size: usize,
        activation_function: &str,
    ) -> UnitParams {
        let weights_dim = (output_size, input_size);

        let weights = UnitParams::generate_new_weights(input_size, output_size);
        let biases = vec![0.0; output_size];

        let activation = activation_function.to_string();

        UnitParams::Softmax {
            input_size,
            output_size,
            weights_dim,
            weights,
            biases,
            activation,
        }
    }

    fn generate_new_weights(input_size: usize, output_size: usize) -> Vec<f32> {
        let scale = f32::sqrt(6.0 / (input_size + output_size) as f32);
        let initial_weights: Array1<f32> =
            Array1::random(output_size * input_size, Uniform::new(-scale, scale));

        initial_weights.to_vec()
    }
}
