// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::penalty::{
        l2_penalty::L2PenaltyUnit, PenaltyBuilder, PenaltyContainer, PenaltyRef, PenaltyType,
        PenaltyUnit,
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

impl<'a> PenaltyBuilder<'a> for L2PenaltyBuilder {
    fn create_first(&self, weights: &NodeRef<'a>) -> PenaltyContainer<'a> {
        let mut unit = L2PenaltyUnit::new(self.alpha);
        unit.add_parameter_input(weights);

        PenaltyContainer::new(unit)
    }

    fn create_new(&self, prev: &PenaltyRef<'a>, weights: &NodeRef<'a>) -> PenaltyContainer<'a> {
        let mut unit = L2PenaltyUnit::new(self.alpha);
        unit.add_parameter_input(weights);
        unit.add_penalty_input(prev);

        PenaltyContainer::new(unit)
    }

    fn get_associated_type(&self) -> PenaltyType {
        PenaltyType::L2 { alpha: self.alpha }
    }
}
