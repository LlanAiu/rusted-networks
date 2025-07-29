// builtin

// external
use serde::{Deserialize, Serialize};

// internal

#[derive(Serialize, Deserialize)]
pub struct LearningParams {
    pub learning_rate: f32,
}
