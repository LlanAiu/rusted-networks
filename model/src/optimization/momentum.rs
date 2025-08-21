// builtin

// external

// internal

use crate::data::{data_container::DataContainer, Data};

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
