// builtin

// external
use serde::{Deserialize, Serialize};

// internal

#[derive(Serialize, Deserialize)]
pub struct BatchNormParams {
    dim: Vec<usize>,
    mean: Vec<f32>,
    variance: Vec<f32>,
    decay: f32,
}

// TODO:
impl BatchNormParams {}
