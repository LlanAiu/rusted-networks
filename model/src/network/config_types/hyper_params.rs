// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::optimization::{
    batch_norm::NormalizationType, learning_decay::LearningDecayType, momentum::DescentType,
};

#[derive(Serialize, Deserialize)]
pub struct HyperParams {
    decay_type: LearningDecayType,
    descent_type: DescentType,
    normalization_type: NormalizationType,
}

impl HyperParams {
    pub fn new(
        decay_type: LearningDecayType,
        descent_type: DescentType,
        normalization_type: NormalizationType,
    ) -> HyperParams {
        HyperParams {
            decay_type,
            descent_type,
            normalization_type,
        }
    }

    pub fn decay_type(&self) -> &LearningDecayType {
        &self.decay_type
    }

    pub fn descent_type(&self) -> &DescentType {
        &self.descent_type
    }

    pub fn normalization_type(&self) -> &NormalizationType {
        &self.normalization_type
    }
}
