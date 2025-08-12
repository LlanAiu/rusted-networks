// builtin

// external
use serde::{Deserialize, Serialize};

// internal

#[derive(Serialize, Deserialize)]
pub struct HyperParams {
    learning_rate: f32,
}

impl HyperParams {
    pub fn new(learning_rate: f32) -> HyperParams {
        HyperParams { learning_rate }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
