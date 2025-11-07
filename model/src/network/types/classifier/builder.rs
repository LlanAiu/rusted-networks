// builtin

// external

// internal
use crate::{
    network::{
        config_types::unit_params::UnitParams,
        types::classifier::{config::ClassifierConfig, ClassifierNetwork},
    },
    node::NodeRef,
    optimization::{learning_decay::LearningDecayType, momentum::DescentType},
    regularization::penalty::{PenaltyConfig, PenaltyContainer},
    unit::{
        types::{
            input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit,
            softmax_unit::SoftmaxUnit,
        },
        UnitContainer, UnitRef,
    },
};

pub fn build_from_config<'a>(config: ClassifierConfig) -> ClassifierNetwork<'a> {
    let decay_type: &LearningDecayType = config.params().decay_type();
    let descent_type: &DescentType = config.params().descent_type();
    let penalty_config: PenaltyConfig = config.regularization().get_config();

    let input: UnitContainer<InputUnit> =
        UnitContainer::new(InputUnit::from_config(config.input()));

    let (hidden, prev_ref, prev_penalty) =
        build_hidden_units(&config, input.get_ref(), &penalty_config);

    let (inference, inference_penalty) =
        build_inference(&config, &penalty_config, prev_penalty, prev_ref);

    let loss: UnitContainer<LossUnit> = build_loss(&config, &inference, &inference_penalty);

    ClassifierNetwork {
        input,
        hidden,
        inference,
        loss,
        penalty_type: penalty_config.get_type(),
        decay_type: decay_type.clone(),
        descent_type: descent_type.clone(),
        time_step: config.timestep(),
    }
}

fn build_hidden_units<'a>(
    config: &ClassifierConfig,
    input_ref: UnitRef<'a>,
    penalty_config: &PenaltyConfig<'a>,
) -> (
    Vec<UnitContainer<'a, LinearUnit<'a>>>,
    UnitRef<'a>,
    Option<PenaltyContainer<'a>>,
) {
    let decay_type: &LearningDecayType = config.params().decay_type();
    let descent_type: &DescentType = config.params().descent_type();
    let hidden_len: usize = config.units().len();
    let units: &Vec<UnitParams> = config.units();

    let mut prev_ref: UnitRef = input_ref;
    let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();
    let mut prev_penalty: Option<PenaltyContainer> = None;

    for i in 0..(hidden_len - 1) {
        let hidden_config: &UnitParams = units.get(i).unwrap();
        let hidden_unit: UnitContainer<LinearUnit> = UnitContainer::new(LinearUnit::from_config(
            hidden_config,
            decay_type.clone(),
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
    config: &ClassifierConfig,
    penalty_config: &PenaltyConfig<'a>,
    prev_penalty: Option<PenaltyContainer<'a>>,
    prev_ref: UnitRef<'a>,
) -> (UnitContainer<'a, SoftmaxUnit<'a>>, PenaltyContainer<'a>) {
    let decay_type: &LearningDecayType = config.params().decay_type();
    let descent_type: &DescentType = config.params().descent_type();
    let hidden_len: usize = config.units().len();
    let inference_config = config.units().get(hidden_len - 1).unwrap();

    let inference: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::from_config(
        inference_config,
        decay_type.clone(),
        descent_type.clone(),
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
    config: &ClassifierConfig,
    inference: &UnitContainer<'a, SoftmaxUnit<'a>>,
    inference_penalty: &PenaltyContainer<'a>,
) -> UnitContainer<'a, LossUnit<'a>> {
    let loss: UnitContainer<LossUnit> = UnitContainer::new(LossUnit::from_config(config.loss()));
    loss.add_input(&inference);
    loss.borrow()
        .add_regularization_node(&inference_penalty.get_ref());

    loss
}
