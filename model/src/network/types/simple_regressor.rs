// builtin
use std::{cell::RefCell, io::Result, rc::Rc};

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::{
        config_types::{
            hyper_params::HyperParams, input_params::InputParams, loss_params::LossParams,
            unit_params::UnitParams,
        },
        types::simple_regressor::config::RegressorConfig,
        Network,
    },
    regularization::penalty::{l2_penalty::L2PenaltyUnit, PenaltyRef},
    unit::{
        types::{input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit},
        Unit, UnitContainer,
    },
};
pub mod config;

pub struct SimpleRegressorNetwork<'a> {
    input: UnitContainer<'a, InputUnit<'a>>,
    hidden: Vec<UnitContainer<'a, LinearUnit<'a>>>,
    inference: UnitContainer<'a, LinearUnit<'a>>,
    loss: UnitContainer<'a, LossUnit<'a>>,
    learning_rate: f32,
    alpha: f32,
}

impl<'a> SimpleRegressorNetwork<'a> {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        learning_rate: f32,
        alpha: f32,
    ) -> SimpleRegressorNetwork<'a> {
        let config: RegressorConfig =
            RegressorConfig::new(input_size, output_size, hidden_sizes, learning_rate, alpha);
        SimpleRegressorNetwork::from_config(config)
    }

    pub fn load_from_file(path: &str) -> SimpleRegressorNetwork<'a> {
        let config: RegressorConfig = RegressorConfig::load_from_file(path).unwrap();
        SimpleRegressorNetwork::from_config(config)
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let input = InputParams::from_unit(&self.input);

        let loss = LossParams::from_unit(&self.loss);

        let mut units: Vec<UnitParams> = Vec::new();

        for unit in &self.hidden {
            units.push(UnitParams::from_linear_unit(unit));
        }
        units.push(UnitParams::from_linear_unit(&self.inference));

        let learning = HyperParams {
            learning_rate: self.learning_rate,
            reg_alpha: self.alpha,
        };

        let config: RegressorConfig = RegressorConfig::from_params(input, units, loss, learning);

        config.save_to_file(path)
    }

    fn from_config(config: RegressorConfig) -> SimpleRegressorNetwork<'a> {
        let learning_rate: f32 = config.params().learning_rate;
        let alpha: f32 = config.params().reg_alpha;

        let input: UnitContainer<InputUnit> =
            UnitContainer::new(InputUnit::from_config(config.input()));

        let hidden_len = config.units().len();
        let units = config.units();

        let mut prev_ref = input.get_ref();
        let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();

        let mut prev_reg_unit: Option<PenaltyRef> = None;

        for i in 0..(hidden_len - 1) {
            let hidden_config = units.get(i).unwrap();
            let hidden_unit: UnitContainer<LinearUnit> =
                UnitContainer::new(LinearUnit::from_config(hidden_config, learning_rate));

            let reg_unit: PenaltyRef = Rc::new(RefCell::new(L2PenaltyUnit::new(alpha)));
            reg_unit
                .borrow_mut()
                .add_parameter_input(hidden_unit.borrow().get_weights_ref());

            if let Option::Some(prev_reg) = prev_reg_unit {
                reg_unit.borrow_mut().add_penalty_input(&prev_reg);
            }

            hidden_unit.add_input_ref(&prev_ref);
            prev_reg_unit = Option::Some(reg_unit);
            prev_ref = hidden_unit.get_ref();
            hidden.push(hidden_unit);
        }

        let inference_config = units.get(hidden_len - 1).unwrap();
        let inference: UnitContainer<LinearUnit> =
            UnitContainer::new(LinearUnit::from_config(inference_config, learning_rate));

        let inference_reg: PenaltyRef = Rc::new(RefCell::new(L2PenaltyUnit::new(alpha)));

        inference_reg
            .borrow_mut()
            .add_parameter_input(inference.borrow().get_weights_ref());
        if let Option::Some(prev_reg) = prev_reg_unit {
            inference_reg.borrow_mut().add_penalty_input(&prev_reg);
        }
        inference.add_input_ref(&prev_ref);

        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::from_config(config.loss()));

        loss.add_input(&inference);
        loss.borrow().add_regularization_node(&inference_reg);

        SimpleRegressorNetwork {
            input,
            hidden,
            inference,
            loss,
            learning_rate,
            alpha,
        }
    }
}

impl Network for SimpleRegressorNetwork<'_> {
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
