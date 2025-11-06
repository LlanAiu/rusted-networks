// builtin

// external

// internal

use core::panic;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum NetworkMode {
    Inference,
    Train,
    None,
}

#[derive(Clone)]
pub enum UnitMaskType {
    None,
    Dropout { keep_probability: f32 },
}

impl UnitMaskType {
    pub fn from_keep_probability(probability: f32) -> UnitMaskType {
        if probability >= 1.0 {
            return UnitMaskType::None;
        } else if probability <= 0.0 {
            panic!("[MASK] Invalid mask probability - must be in (0, 1]!");
        }
        UnitMaskType::Dropout {
            keep_probability: probability,
        }
    }

    pub fn probability(&self) -> f32 {
        match self {
            UnitMaskType::None => 1.0,
            UnitMaskType::Dropout {
                keep_probability: probability,
            } => *probability,
        }
    }
}
