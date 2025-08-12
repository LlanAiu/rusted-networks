// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::penalty::{
        l1_penalty::L1PenaltyUnit, PenaltyBuilder, PenaltyContainer, PenaltyRef, PenaltyType,
        PenaltyUnit,
    },
};

pub struct L1PenaltyBuilder {
    alpha: f32,
}

impl L1PenaltyBuilder {
    pub fn new(alpha: f32) -> L1PenaltyBuilder {
        L1PenaltyBuilder { alpha }
    }
}

impl<'a> PenaltyBuilder<'a> for L1PenaltyBuilder {
    fn create_first(&self, weights: &NodeRef<'a>) -> PenaltyContainer<'a> {
        let mut unit = L1PenaltyUnit::new(self.alpha);
        unit.add_parameter_input(weights);

        PenaltyContainer::new(unit)
    }

    fn create_new(&self, prev: &PenaltyRef<'a>, weights: &NodeRef<'a>) -> PenaltyContainer<'a> {
        let mut unit = L1PenaltyUnit::new(self.alpha);
        unit.add_parameter_input(weights);
        unit.add_penalty_input(prev);

        PenaltyContainer::new(unit)
    }

    fn get_associated_type(&self) -> PenaltyType {
        PenaltyType::L1 { alpha: self.alpha }
    }
}
