// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::norm_penalty::{
        no_penalty::NullPenaltyUnit, NormPenaltyBuilder, NormPenaltyContainer, NormPenaltyRef,
        NormPenaltyType,
    },
};

pub struct NullBuilder;

impl<'a> NormPenaltyBuilder<'a> for NullBuilder {
    fn create_first(&self, _node: &NodeRef<'a>) -> NormPenaltyContainer<'a> {
        let unit = NullPenaltyUnit::new();

        NormPenaltyContainer::new(unit)
    }

    fn create_new(
        &self,
        _prev: &NormPenaltyRef<'a>,
        _weights: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a> {
        let unit = NullPenaltyUnit::new();

        NormPenaltyContainer::new(unit)
    }

    fn get_associated_type(&self) -> NormPenaltyType {
        NormPenaltyType::None
    }
}
