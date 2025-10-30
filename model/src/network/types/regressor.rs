// builtin
use std::io::Result;

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::{
        config_types::Config,
        types::regressor::{builder::build_from_config, config::RegressorConfig},
        Network,
    },
    optimization::{learning_decay::LearningDecayType, momentum::DescentType},
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
    penalty_type: PenaltyType,
    with_dropout: bool,
    decay_type: LearningDecayType,
    descent_type: DescentType,
    time_step: usize,
}

impl<'a> RegressorNetwork<'a> {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        penalty_config: PenaltyConfig,
        with_dropout: bool,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> RegressorNetwork<'a> {
        let config: RegressorConfig = RegressorConfig::new(
            input_size,
            output_size,
            hidden_sizes,
            penalty_config,
            with_dropout,
            decay_type,
            descent_type,
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

    fn train(&mut self, input: DataContainer, response: DataContainer) {
        self.input.borrow().set_input_data(input);
        self.loss.borrow().set_expected_response(response);

        let loss_ref = self.loss.borrow();
        let loss_node = loss_ref.get_output_node();

        loss_node.borrow_mut().apply_operation();

        loss_node.borrow_mut().add_gradient(&DataContainer::one());
        loss_node.borrow_mut().apply_jacobian();
    }

    fn create_config(&self) -> Config {
        let regressor_config = RegressorConfig::to_config(self);
        Config::Regressor(regressor_config)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Array1};
    use rand::random_range;

    use crate::{
        data::{data_container::DataContainer, Data},
        network::{types::regressor::RegressorNetwork, Network},
        optimization::{learning_decay::LearningDecayType, momentum::DescentType},
        regularization::penalty::{l2_penalty::builder::L2PenaltyBuilder, PenaltyConfig},
    };

    #[test]
    fn regressor_load_test() {
        let regressor: RegressorNetwork =
            RegressorNetwork::load_from_file("test/quadratic_training.json");

        let test_arr: Array1<f32> = arr1(&[2.0]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = regressor.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[3.0]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = regressor.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);

        let test_arr3: Array1<f32> = arr1(&[4.0]);
        let after_data3 = DataContainer::Inference(Data::VectorF32(test_arr3.clone()));
        let after_output3 = regressor.predict(after_data3);
        println!("Loaded output 3: {:?}", after_output3);
    }

    #[test]
    fn regressor_regularization_test() {
        let config: PenaltyConfig = PenaltyConfig::new(L2PenaltyBuilder::new(0.2));

        let mut regressor: RegressorNetwork = RegressorNetwork::new(
            vec![1],
            vec![1],
            vec![6, 3],
            config,
            false,
            LearningDecayType::constant(0.005),
            DescentType::Base,
        );

        for _i in 0..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 0..8 {
                let x: f32 = random_range(1.0..4.0);

                inputs.push(Data::VectorF32(arr1(&[x])));
                responses.push(Data::VectorF32(arr1(&[x * x])));
            }

            let input = DataContainer::Batch(inputs);
            let response = DataContainer::Batch(responses);

            regressor.train(input, response);
        }

        let test_arr: Array1<f32> = arr1(&[2.0]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = regressor.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[3.0]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = regressor.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);

        regressor
            .save_to_file("test/regressor_test.json")
            .expect("Save Failed");
    }
}
