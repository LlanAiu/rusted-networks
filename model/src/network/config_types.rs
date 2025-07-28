// builtin

// external
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::types::{
        binary_classifier::config::BinaryClassifierConfig,
        simple_classifier::config::ClassifierConfig, simple_regressor::config::RegressorConfig,
    },
    unit::{
        types::{
            input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit,
            softmax_unit::SoftmaxUnit,
        },
        UnitContainer,
    },
};

#[derive(Serialize, Deserialize)]
pub enum Config {
    BinaryClassifier(BinaryClassifierConfig),
    Classifier(ClassifierConfig),
    Regressor(RegressorConfig),
}

#[derive(Serialize, Deserialize)]
pub struct InputParams {
    pub input_size: Vec<usize>,
}

impl InputParams {
    pub fn from_unit<'a>(unit: &UnitContainer<'a, InputUnit<'a>>) -> InputParams {
        InputParams {
            input_size: unit.borrow().get_input_size().to_vec(),
        }
    }
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
}

#[derive(Serialize, Deserialize)]
pub struct LossParams {
    pub loss_type: String,
    pub output_size: Vec<usize>,
}

impl LossParams {
    pub fn from_unit<'a>(unit: &UnitContainer<'a, LossUnit<'a>>) -> LossParams {
        let loss_type = unit.borrow().get_loss_type().to_string();
        let output_size = unit.borrow().get_output_size().to_vec();

        LossParams {
            loss_type,
            output_size,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct LearningParams {
    pub learning_rate: f32,
}
