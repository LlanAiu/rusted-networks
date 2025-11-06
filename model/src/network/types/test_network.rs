// builtin

// external

// internal
use crate::{
    optimization::{learning_decay::LearningDecayType, momentum::DescentType},
    regularization::{dropout::UnitMaskType, penalty::PenaltyConfig},
    unit::{
        types::{input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit},
        UnitContainer,
    },
};

pub struct TestNetwork<'a> {
    _input: UnitContainer<'a, InputUnit<'a>>,
    _inference: UnitContainer<'a, LinearUnit<'a>>,
    _loss: UnitContainer<'a, LossUnit<'a>>,
}

impl<'a> TestNetwork<'a> {
    pub fn new(
        input_size: usize,
        output_size: usize,
        decay_type: LearningDecayType,
        penalty: PenaltyConfig<'a>,
    ) -> TestNetwork<'a> {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(vec![input_size]));
        let inference: UnitContainer<LinearUnit> = UnitContainer::new(LinearUnit::new(
            "none",
            input_size,
            output_size,
            decay_type,
            DescentType::Base,
            UnitMaskType::None,
            true,
        ));

        let reg_unit = penalty.create_first(inference.borrow().get_weights_ref());

        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(vec![output_size], "base_cross_entropy"));

        loss.add_input(&inference);
        inference.add_input(&input);

        loss.borrow_mut()
            .add_regularization_node(&reg_unit.get_ref());

        TestNetwork {
            _input: input,
            _inference: inference,
            _loss: loss,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        network::types::test_network::TestNetwork,
        optimization::learning_decay::LearningDecayType,
        regularization::penalty::{l2_penalty::builder::L2PenaltyBuilder, PenaltyConfig},
    };

    #[test]
    fn create_test() {
        let builder: L2PenaltyBuilder = L2PenaltyBuilder::new(0.2);
        let penalty_config = PenaltyConfig::new(builder);
        let _net: TestNetwork =
            TestNetwork::new(3, 4, LearningDecayType::constant(0.001), penalty_config);
    }
}
