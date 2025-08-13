// builtin
use std::io::Result;

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::{
        types::regressor::{builder::build_from_config, config::RegressorConfig},
        Network,
    },
    regularization::penalty::{PenaltyConfig, PenaltyType},
    unit::{
        types::{input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit},
        Unit, UnitContainer,
    },
};
pub mod builder;
pub mod config;

pub struct RegressorNetwork<'a> {
    input: UnitContainer<'a, InputUnit<'a>>,
    hidden: Vec<UnitContainer<'a, LinearUnit<'a>>>,
    inference: UnitContainer<'a, LinearUnit<'a>>,
    loss: UnitContainer<'a, LossUnit<'a>>,
    learning_rate: f32,
    penalty_type: PenaltyType,
    with_dropout: bool,
}

impl<'a> RegressorNetwork<'a> {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        learning_rate: f32,
        penalty_config: PenaltyConfig,
        with_dropout: bool,
    ) -> RegressorNetwork<'a> {
        let config: RegressorConfig = RegressorConfig::new(
            input_size,
            output_size,
            hidden_sizes,
            learning_rate,
            penalty_config,
            with_dropout,
        );
        RegressorNetwork::from_config(config)
    }

    pub fn load_from_file(path: &str) -> RegressorNetwork<'a> {
        let config: RegressorConfig = RegressorConfig::load_from_file(path).unwrap();
        RegressorNetwork::from_config(config)
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let config: RegressorConfig = RegressorConfig::to_config(self);
        config.save_to_file(path)
    }

    fn from_config(config: RegressorConfig) -> RegressorNetwork<'a> {
        build_from_config(config)
    }
}

impl Network for RegressorNetwork<'_> {
    fn predict(&self, input: DataContainer) -> DataContainer {
        self.input.borrow_mut().set_input_data(input);

        let inference_ref = self.inference.borrow();
        let inference_node = inference_ref.get_output_node();

        inference_node.borrow_mut().apply_operation();

        let output = inference_node.borrow_mut().get_data();

        output
    }

    fn train(&self, input: DataContainer, response: DataContainer) {
        self.input.borrow().set_input_data(input);
        self.loss.borrow().set_expected_response(response);

        let loss_ref = self.loss.borrow();
        let loss_node = loss_ref.get_output_node();

        loss_node.borrow_mut().apply_operation();

        loss_node.borrow_mut().add_gradient(&DataContainer::one());
        loss_node.borrow_mut().apply_jacobian();
    }
}
