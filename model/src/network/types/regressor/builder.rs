// builtin

// external

// internal
use crate::{
    network::{
        config_types::unit_params::UnitParams,
        types::regressor::{config::RegressorConfig, RegressorNetwork},
    },
    node::NodeRef,
    optimization::{learning_decay::LearningDecay, momentum::DescentType},
    regularization::penalty::{PenaltyConfig, PenaltyContainer},
    unit::{
        types::{input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit},
        UnitContainer, UnitRef,
    },
};

pub fn build_from_config<'a>(config: RegressorConfig) -> RegressorNetwork<'a> {
    let learning_decay: LearningDecay = config.params().learning_decay().clone();
    let descent_type: DescentType = config.params().descent().to_type();
    let penalty_config: PenaltyConfig = config.regularization().get_config();

    let input: UnitContainer<InputUnit> =
        UnitContainer::new(InputUnit::from_config(config.input()));

    let (hidden, prev_ref, prev_penalty) =
        build_hidden_units(&config, input.get_ref(), &penalty_config);

    let (inference, inference_penalty) =
        build_inference(&config, &penalty_config, prev_penalty, prev_ref);

    let loss: UnitContainer<LossUnit> = build_loss(&config, &inference, &inference_penalty);

    RegressorNetwork {
        input,
        hidden,
        inference,
        loss,
        learning_decay,
        penalty_type: penalty_config.get_type(),
        with_dropout: config.regularization().is_dropout_enabled(),
        descent_type,
    }
}

fn build_hidden_units<'a>(
    config: &RegressorConfig,
    input_ref: UnitRef<'a>,
    penalty_config: &PenaltyConfig<'a>,
) -> (
    Vec<UnitContainer<'a, LinearUnit<'a>>>,
    UnitRef<'a>,
    Option<PenaltyContainer<'a>>,
) {
    let learning_rate: f32 = *config.params().learning_decay().get_learning_rate_f32();
    let descent_type: DescentType = config.params().descent().to_type();
    let hidden_len: usize = config.units().len();
    let units: &Vec<UnitParams> = config.units();

    let mut prev_ref: UnitRef = input_ref;
    let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();
    let mut prev_penalty: Option<PenaltyContainer> = None;

    for i in 0..(hidden_len - 1) {
        let hidden_config: &UnitParams = units.get(i).unwrap();
        let hidden_unit: UnitContainer<LinearUnit> = UnitContainer::new(LinearUnit::from_config(
            hidden_config,
            learning_rate,
            descent_type.clone(),
        ));

        let penalty: PenaltyContainer = build_penalty(
            penalty_config,
            prev_penalty,
            hidden_unit.borrow().get_weights_ref(),
        );

        hidden_unit.add_input_ref(&prev_ref);
        prev_penalty = Option::Some(penalty);
        prev_ref = hidden_unit.get_ref();
        hidden.push(hidden_unit);
    }

    (hidden, prev_ref, prev_penalty)
}

fn build_penalty<'a>(
    penalty_config: &PenaltyConfig<'a>,
    prev_penalty: Option<PenaltyContainer<'a>>,
    parameter: &NodeRef<'a>,
) -> PenaltyContainer<'a> {
    if let Option::Some(prev) = &prev_penalty {
        penalty_config.create_new(&prev.get_ref(), parameter)
    } else {
        penalty_config.create_first(parameter)
    }
}

fn build_inference<'a>(
    config: &RegressorConfig,
    penalty_config: &PenaltyConfig<'a>,
    prev_penalty: Option<PenaltyContainer<'a>>,
    prev_ref: UnitRef<'a>,
) -> (UnitContainer<'a, LinearUnit<'a>>, PenaltyContainer<'a>) {
    let learning_rate: f32 = *config.params().learning_decay().get_learning_rate_f32();
    let descent_type: DescentType = config.params().descent().to_type();
    let hidden_len: usize = config.units().len();
    let inference_config = config.units().get(hidden_len - 1).unwrap();

    let inference: UnitContainer<LinearUnit> = UnitContainer::new(LinearUnit::from_config(
        inference_config,
        learning_rate,
        descent_type,
    ));

    let inference_penalty: PenaltyContainer = build_penalty(
        &penalty_config,
        prev_penalty,
        inference.borrow().get_weights_ref(),
    );

    inference.add_input_ref(&prev_ref);

    (inference, inference_penalty)
}

fn build_loss<'a>(
    config: &RegressorConfig,
    inference: &UnitContainer<'a, LinearUnit<'a>>,
    inference_penalty: &PenaltyContainer<'a>,
) -> UnitContainer<'a, LossUnit<'a>> {
    let loss: UnitContainer<LossUnit> = UnitContainer::new(LossUnit::from_config(config.loss()));
    loss.add_input(&inference);
    loss.borrow()
        .add_regularization_node(&inference_penalty.get_ref());

    loss
}
