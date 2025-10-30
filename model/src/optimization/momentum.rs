// builtin

// external
use serde::{Deserialize, Serialize};

// internal

#[derive(Serialize, Deserialize, Clone)]
pub enum DescentType {
    Base,
    Momentum { decay: f32 },
    Nesterov { decay: f32 },
}

impl DescentType {
    pub fn none() -> DescentType {
        DescentType::Base
    }

    pub fn momentum(decay: f32) -> DescentType {
        DescentType::Momentum { decay }
    }

    pub fn nesterov(decay: f32) -> DescentType {
        DescentType::Nesterov { decay }
    }
}

#[derive(Serialize, Deserialize)]
pub struct MomentumParams {
    momentum: Vec<f32>,
}

impl MomentumParams {
    pub fn new(momentum: Vec<f32>) -> MomentumParams {
        MomentumParams { momentum }
    }

    pub fn null() -> MomentumParams {
        MomentumParams {
            momentum: Vec::new(),
        }
    }

    pub fn get_momentum(&self) -> Vec<f32> {
        self.momentum.clone()
    }
}
