// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::regularization::norm_penalty::{
    l2_penalty::builder::L2PenaltyBuilder, NormPenaltyBuilder, NormPenaltyConfig, NormPenaltyType,
};

#[derive(Serialize, Deserialize)]
pub struct RegularizationParams {
    norm_penalty: NormPenaltyType,
    with_dropout: bool,
}

impl RegularizationParams {
    pub fn new(norm_type: NormPenaltyType, with_dropout: bool) -> RegularizationParams {
        RegularizationParams {
            norm_penalty: norm_type,
            with_dropout,
        }
    }

    pub fn from_builder<'a>(
        builder: &Box<dyn NormPenaltyBuilder<'a> + 'a>,
        with_dropout: bool,
    ) -> RegularizationParams {
        let penalty_type: NormPenaltyType = builder.get_associated_type();

        RegularizationParams::new(penalty_type, with_dropout)
    }

    pub fn get_config<'a>(&self) -> NormPenaltyConfig<'a> {
        match self.norm_penalty {
            NormPenaltyType::L2 { alpha } => {
                let builder = L2PenaltyBuilder::new(alpha);
                NormPenaltyConfig::new(builder)
            }
            NormPenaltyType::None => NormPenaltyConfig::none(),
        }
    }
}
