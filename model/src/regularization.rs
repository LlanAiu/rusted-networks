// builtin

// external

// internal
pub mod norm_penalty;

pub struct RegularizationConfig {
    reg_type: RegularizationType,
}

pub enum RegularizationType {
    L2Penalty,
}
