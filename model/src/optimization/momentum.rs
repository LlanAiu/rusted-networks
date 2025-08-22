// builtin

// external

use serde::{Deserialize, Serialize};

// internal
use crate::data::{data_container::DataContainer, Data};

#[derive(Clone)]
pub enum DescentType {
    Base,
    Momentum { decay: DataContainer },
    Nesterov { decay: DataContainer },
}

impl DescentType {
    pub fn momentum(decay: f32) -> DescentType {
        DescentType::Momentum {
            decay: DataContainer::Parameter(Data::ScalarF32(decay)),
        }
    }

    pub fn nesterov(decay: f32) -> DescentType {
        DescentType::Nesterov {
            decay: DataContainer::Parameter(Data::ScalarF32(decay)),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum DescentParams {
    Base,
    Momentum { decay: f32 },
    Nesterov { decay: f32 },
}

impl DescentParams {
    pub fn from_type(descent_type: &DescentType) -> DescentParams {
        match descent_type {
            DescentType::Base => DescentParams::Base,
            DescentType::Momentum { decay } => {
                if let DataContainer::Parameter(Data::ScalarF32(val)) = decay {
                    return DescentParams::Momentum { decay: *val };
                }
                DescentParams::Momentum { decay: 0.0 }
            }
            DescentType::Nesterov { decay } => {
                if let DataContainer::Parameter(Data::ScalarF32(val)) = decay {
                    return DescentParams::Nesterov { decay: *val };
                }
                DescentParams::Nesterov { decay: 0.0 }
            }
        }
    }

    pub fn to_type(&self) -> DescentType {
        match self {
            DescentParams::Base => DescentType::Base,
            DescentParams::Momentum { decay } => DescentType::momentum(*decay),
            DescentParams::Nesterov { decay } => DescentType::nesterov(*decay),
        }
    }
}
