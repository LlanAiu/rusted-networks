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

#[derive(Clone)]
pub enum NetworkMaskType {
    None,
    Dropout {
        input_keep_p: f32,
        hidden_keep_p: f32,
    },
}

impl NetworkMaskType {
    pub fn from_probabilities(input_keep_p: f32, hidden_keep_p: f32) -> NetworkMaskType {
        if input_keep_p >= 1.0 && hidden_keep_p >= 1.0 {
            return NetworkMaskType::None;
        } else if input_keep_p <= 0.0 || hidden_keep_p <= 0.0 {
            panic!("[MASK] Invalid mask probability - must be in (0, 1]!");
        }
        NetworkMaskType::Dropout {
            input_keep_p,
            hidden_keep_p,
        }
    }

    pub fn input_probability(&self) -> f32 {
        match self {
            NetworkMaskType::None => 1.0,
            NetworkMaskType::Dropout { input_keep_p, .. } => *input_keep_p,
        }
    }

    pub fn hidden_probability(&self) -> f32 {
        match self {
            NetworkMaskType::None => 1.0,
            NetworkMaskType::Dropout { hidden_keep_p, .. } => *hidden_keep_p,
        }
    }
}
