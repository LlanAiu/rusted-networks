// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::norm_penalty::{
        l2_penalty::L2PenaltyUnit, NormPenaltyBuilder, NormPenaltyContainer, NormPenaltyRef,
        NormPenaltyUnit,
    },
};

pub struct L2PenaltyBuilder {
    scale: f32,
}

impl L2PenaltyBuilder {
    pub fn new(scale: f32) -> L2PenaltyBuilder {
        L2PenaltyBuilder { scale }
    }
}

impl<'a> NormPenaltyBuilder<'a, L2PenaltyUnit<'a>> for L2PenaltyBuilder {
    fn create_first(&self, weights: &NodeRef<'a>) -> NormPenaltyContainer<'a, L2PenaltyUnit<'a>> {
        let mut unit = L2PenaltyUnit::new(self.scale);
        unit.add_weight_input(weights);

        NormPenaltyContainer::new(unit)
    }

    fn create_new(
        &self,
        prev: &NormPenaltyRef<'a>,
        weights: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a, L2PenaltyUnit<'a>> {
        let mut unit = L2PenaltyUnit::new(self.scale);
        unit.add_weight_input(weights);
        unit.add_penalty_input(prev);

        NormPenaltyContainer::new(unit)
    }

    fn create_last(
        &self,
        weights: &NodeRef<'a>,
        prev: &NormPenaltyRef<'a>,
        loss_sum_node: &NodeRef<'a>,
    ) -> NormPenaltyContainer<'a, L2PenaltyUnit<'a>> {
        let mut unit = L2PenaltyUnit::new(self.scale);
        unit.add_weight_input(weights);
        unit.add_penalty_input(prev);

        loss_sum_node
            .borrow_mut()
            .add_input(loss_sum_node, unit.get_output_ref());

        NormPenaltyContainer::new(unit)
    }
}
