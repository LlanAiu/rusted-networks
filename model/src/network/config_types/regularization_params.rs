// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::regularization::penalty::{
    l1_penalty::builder::L1PenaltyBuilder, l2_penalty::builder::L2PenaltyBuilder, PenaltyBuilder,
    PenaltyConfig, PenaltyType,
};

#[derive(Serialize, Deserialize)]
pub struct RegularizationParams {
    norm_penalty: PenaltyType,
}

impl RegularizationParams {
    pub fn new(norm_type: PenaltyType) -> RegularizationParams {
        RegularizationParams {
            norm_penalty: norm_type,
        }
    }

    pub fn from_builder<'a>(builder: &Box<dyn PenaltyBuilder<'a> + 'a>) -> RegularizationParams {
        let penalty_type: PenaltyType = builder.get_associated_type();

        RegularizationParams::new(penalty_type)
    }

    pub fn get_config<'a>(&self) -> PenaltyConfig<'a> {
        match self.norm_penalty {
            PenaltyType::L1 { alpha } => {
                let builder = L1PenaltyBuilder::new(alpha);
                PenaltyConfig::new(builder)
            }
            PenaltyType::L2 { alpha } => {
                let builder = L2PenaltyBuilder::new(alpha);
                PenaltyConfig::new(builder)
            }
            PenaltyType::None => PenaltyConfig::none(),
        }
    }
}
