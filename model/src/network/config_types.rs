// builtin

// external
use serde::{Deserialize, Serialize};

// internal

#[derive(Serialize, Deserialize)]
pub struct InputParams {
    pub input_size: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "unit_type")]
pub enum UnitParams {
    Linear {
        input_size: usize,
        output_size: usize,
        weights_dim: (usize, usize),
        weights: Vec<usize>,
        biases: Vec<usize>,
        activation: String,
    },
    Softmax {
        input_size: usize,
        output_size: usize,
        weights_dim: (usize, usize),
        weights: Vec<usize>,
        biases: Vec<usize>,
        activation: String,
    },
}

#[derive(Serialize, Deserialize)]
pub struct LossParams {
    pub loss_type: String,
    pub response_size: usize,
}

#[derive(Serialize, Deserialize)]
pub struct LearningParams {
    pub learning_rate: f32,
}
