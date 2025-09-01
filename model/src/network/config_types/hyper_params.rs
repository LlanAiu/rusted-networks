// builtin

// external
use serde::{Deserialize, Serialize};

use crate::optimization::{
    learning_decay::LearningDecay,
    momentum::{DescentParams, DescentType},
};

// internal

#[derive(Serialize, Deserialize)]
pub struct HyperParams {
    learning_decay: LearningDecay,
    descent: DescentParams,
}

impl HyperParams {
    pub fn new(learning_decay: LearningDecay, descent_type: DescentType) -> HyperParams {
        HyperParams {
            learning_decay,
            descent: DescentParams::from_type(&descent_type),
        }
    }

    pub fn learning_decay(&self) -> &LearningDecay {
        &self.learning_decay
    }

    pub fn descent(&self) -> &DescentParams {
        &self.descent
    }
}
