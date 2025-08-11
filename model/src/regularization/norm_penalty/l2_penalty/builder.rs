// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::norm_penalty::{
        l2_penalty::L2PenaltyUnit, NormPenaltyBuilder, NormPenaltyContainer, NormPenaltyRef,
        NormPenaltyType, NormPenaltyUnit,
    },
};

pub struct L2PenaltyBuilder {
    alpha: f32,
}

impl L2PenaltyBuilder {
    pub fn new(alpha: f32) -> L2PenaltyBuilder {
        L2PenaltyBuilder { alpha }
    }
}

impl<'a> NormPenaltyBuilder<'a> for L2PenaltyBuilder {
    fn create_first(&self, weights: &NodeRef<'a>) -> NormPenaltyContainer<'a> {
        let mut unit = L2PenaltyUnit::new(self.alpha);
        unit.add_parameter_input(weights);

        NormPenaltyContainer::new(unit)
    }

    fn create_new(
        &self,
        prev: &NormPenaltyRef<'a>,
        weights: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a> {
        let mut unit = L2PenaltyUnit::new(self.alpha);
        unit.add_parameter_input(weights);
        unit.add_penalty_input(prev);

        NormPenaltyContainer::new(unit)
    }

    fn get_associated_type(&self) -> NormPenaltyType {
        NormPenaltyType::L2 { alpha: self.alpha }
    }
}
