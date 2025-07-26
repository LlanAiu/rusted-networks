// builtin

use ndarray::{Array1, Array2};
// external
use serde::{Deserialize, Serialize};

use crate::data::{data_container::DataContainer, Data};

// internal

#[derive(Serialize, Deserialize)]
pub struct InputParams {
    pub input_size: Vec<usize>,
}

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
}

#[derive(Serialize, Deserialize)]
pub struct LossParams {
    pub loss_type: String,
    pub output_size: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct LearningParams {
    pub learning_rate: f32,
}
