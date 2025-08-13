// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::penalty::{
        no_penalty::NullPenaltyUnit, PenaltyBuilder, PenaltyContainer, PenaltyRef, PenaltyType,
    },
};

pub struct NullBuilder;

impl<'a> PenaltyBuilder<'a> for NullBuilder {
    fn create_first(&self, _node: &NodeRef<'a>) -> PenaltyContainer<'a> {
        let unit = NullPenaltyUnit::new();

        PenaltyContainer::new(unit)
    }

    fn create_new(&self, _prev: &PenaltyRef<'a>, _weights: &NodeRef<'a>) -> PenaltyContainer<'a> {
        let unit = NullPenaltyUnit::new();

        PenaltyContainer::new(unit)
    }

    fn get_associated_type(&self) -> PenaltyType {
        PenaltyType::None
    }
}
