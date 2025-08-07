// builtin

// external
use serde::{Deserialize, Serialize};

// internal

#[derive(Serialize, Deserialize)]
pub struct HyperParams {
    pub learning_rate: f32,
    pub reg_alpha: f32,
}

impl HyperParams {
    pub fn new(learning_rate: f32, reg_alpha: f32) -> HyperParams {
        HyperParams {
            learning_rate,
            reg_alpha,
        }
    }
}
