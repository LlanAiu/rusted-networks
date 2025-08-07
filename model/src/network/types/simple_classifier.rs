// builtin

// external

use std::io::Result;

// internal
use crate::{
    data::data_container::DataContainer,
    network::{
        config_types::{
            hyper_params::HyperParams, input_params::InputParams, loss_params::LossParams,
            unit_params::UnitParams,
        },
        types::simple_classifier::config::ClassifierConfig,
        Network,
    },
    unit::{
        types::{
            input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit,
            softmax_unit::SoftmaxUnit,
        },
        Unit, UnitContainer, UnitRef,
    },
};
pub mod config;

pub struct SimpleClassifierNetwork<'a> {
    input: UnitContainer<'a, InputUnit<'a>>,
    hidden: Vec<UnitContainer<'a, LinearUnit<'a>>>,
    inference: UnitContainer<'a, SoftmaxUnit<'a>>,
    loss: UnitContainer<'a, LossUnit<'a>>,
    learning_rate: f32,
}

impl<'a> SimpleClassifierNetwork<'a> {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        learning_rate: f32,
    ) -> SimpleClassifierNetwork<'a> {
        if input_size.len() != 1 || output_size.len() != 1 {
            panic!("[SIMPLE_CLASSIFIER] Invalid input / output dimensions for network type, expected 1 and 1 but got {} and {}.", input_size.len(), output_size.len());
        }
        let mut prev_width = input_size[0];
        let output_dim = output_size[0];

        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(input_size));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(output_size, "base_cross_entropy"));
        let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();

        let mut prev_unit: UnitRef = input.get_ref();

        for i in 0..hidden_sizes.len() {
            let width = hidden_sizes
                .get(i)
                .expect("[SIMPLE_CLASSIFIER] Failed to get layer width");

            if *width == 0 {
                println!("[SIMPLE_CLASSIFIER] got layer with 0 width, skipping creation...");
                continue;
            }

            let hidden_unit: UnitContainer<LinearUnit> =
                UnitContainer::new(LinearUnit::new("relu", prev_width, *width, learning_rate));

            hidden_unit.add_input_ref(&prev_unit);

            prev_unit = hidden_unit.get_ref();
            prev_width = *width;

            hidden.push(hidden_unit);
        }

        let inference: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new(
            "none",
            prev_width,
            output_dim,
            learning_rate,
        ));
        inference.add_input_ref(&prev_unit);

        loss.add_input(&inference);

        SimpleClassifierNetwork {
            input,
            hidden,
            inference,
            loss,
            learning_rate,
        }
    }

    pub fn load_from_file(path: &str) -> SimpleClassifierNetwork {
        let config: ClassifierConfig = ClassifierConfig::load_from_file(path).unwrap();
        let learning_rate: f32 = config.learning().learning_rate;

        let input: UnitContainer<InputUnit> =
            UnitContainer::new(InputUnit::from_config(config.input()));

        let hidden_len = config.units().len();
        let units = config.units();

        let mut prev_ref = input.get_ref();
        let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();

        for i in 0..(hidden_len - 1) {
            let hidden_config = units.get(i).unwrap();
            let hidden_unit: UnitContainer<LinearUnit> =
                UnitContainer::new(LinearUnit::from_config(hidden_config, learning_rate));

            hidden_unit.add_input_ref(&prev_ref);
            prev_ref = hidden_unit.get_ref();
            hidden.push(hidden_unit);
        }

        let inference_config = units.get(hidden_len - 1).unwrap();
        let inference: UnitContainer<SoftmaxUnit> =
            UnitContainer::new(SoftmaxUnit::from_config(inference_config, learning_rate));

        inference.add_input_ref(&prev_ref);

        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::from_config(config.loss()));

        loss.add_input(&inference);

        SimpleClassifierNetwork {
            input,
            hidden,
            inference,
            loss,
            learning_rate,
        }
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let input = InputParams::from_unit(&self.input);

        let loss = LossParams::from_unit(&self.loss);

        let mut units: Vec<UnitParams> = Vec::new();

        for unit in &self.hidden {
            units.push(UnitParams::from_linear_unit(unit));
        }
        units.push(UnitParams::from_softmax_unit(&self.inference));

        let learning = HyperParams {
            learning_rate: self.learning_rate,
            reg_alpha: 0.0,
        };

        let config: ClassifierConfig = ClassifierConfig::new(input, units, loss, learning);

        config.save_to_file(path)
    }
}

impl Network for SimpleClassifierNetwork<'_> {
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
