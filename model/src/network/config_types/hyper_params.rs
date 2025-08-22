// builtin

// external
use serde::{Deserialize, Serialize};

use crate::optimization::momentum::{DescentParams, DescentType};

// internal

#[derive(Serialize, Deserialize)]
pub struct HyperParams {
    learning_rate: f32,
    descent: DescentParams,
}

impl HyperParams {
    pub fn new(learning_rate: f32, descent_type: DescentType) -> HyperParams {
        HyperParams {
            learning_rate,
            descent: DescentParams::from_type(&descent_type),
        }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn descent(&self) -> &DescentParams {
        &self.descent
    }
}
