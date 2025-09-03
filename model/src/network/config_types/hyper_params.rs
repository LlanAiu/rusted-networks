// builtin

// external
use serde::{Deserialize, Serialize};

use crate::optimization::{learning_decay::LearningDecay, momentum::DescentType};

// internal

#[derive(Serialize, Deserialize)]
pub struct HyperParams {
    learning_decay: LearningDecay,
    descent_type: DescentType,
}

impl HyperParams {
    pub fn new(learning_decay: LearningDecay, descent_type: DescentType) -> HyperParams {
        HyperParams {
            learning_decay,
            descent_type,
        }
    }

    pub fn learning_decay(&self) -> &LearningDecay {
        &self.learning_decay
    }

    pub fn descent(&self) -> &DescentType {
        &self.descent_type
    }
}
