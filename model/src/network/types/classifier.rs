// builtin

// external

use std::io::Result;

// internal
use crate::{
    data::data_container::DataContainer,
    network::{
        config_types::Config,
        types::classifier::{builder::build_from_config, config::ClassifierConfig},
        Network,
    },
    optimization::{learning_decay::LearningDecayType, momentum::DescentType},
    regularization::{
        dropout::NetworkMaskType,
        penalty::{PenaltyConfig, PenaltyType},
    },
    unit::{
        types::{
            input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit,
            softmax_unit::SoftmaxUnit,
        },
        Unit, UnitContainer,
    },
};
pub mod builder;
pub mod config;

pub struct ClassifierNetwork<'a> {
    input: UnitContainer<'a, InputUnit<'a>>,
    hidden: Vec<UnitContainer<'a, LinearUnit<'a>>>,
    inference: UnitContainer<'a, SoftmaxUnit<'a>>,
    loss: UnitContainer<'a, LossUnit<'a>>,
    penalty_type: PenaltyType,
    decay_type: LearningDecayType,
    descent_type: DescentType,
    time_step: usize,
}

impl<'a> ClassifierNetwork<'a> {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        penalty_config: PenaltyConfig,
        mask_type: NetworkMaskType,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> ClassifierNetwork<'a> {
        let config: ClassifierConfig = ClassifierConfig::new(
            input_size,
            output_size,
            hidden_sizes,
            penalty_config,
            mask_type,
            decay_type,
            descent_type,
        );

        ClassifierNetwork::from_config(config)
    }

    pub fn load_from_file(path: &str) -> ClassifierNetwork<'a> {
        let config: ClassifierConfig = ClassifierConfig::load_from_file(path).unwrap();
        ClassifierNetwork::from_config(config)
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let config: ClassifierConfig = ClassifierConfig::from_network(&self);
        config.save_to_file(path)
    }

    fn from_config(config: ClassifierConfig) -> ClassifierNetwork<'a> {
        build_from_config(config)
    }
}

impl Network for ClassifierNetwork<'_> {
    fn predict(&self, input: DataContainer) -> DataContainer {
        self.input.borrow_mut().set_input_data(input);

        let inference_ref = self.inference.borrow();
        let inference_node = inference_ref.get_output_node();

        inference_node.borrow_mut().apply_operation();

        let output = inference_node.borrow_mut().get_data();

        output
    }

    fn train(&mut self, input: DataContainer, response: DataContainer) {
        self.time_step += 1;

        self.input.borrow().set_input_data(input);
        self.loss.borrow().set_expected_response(response);

        let loss_ref = self.loss.borrow();
        let loss_node = loss_ref.get_output_node();

        loss_node.borrow_mut().apply_operation();

        loss_node.borrow_mut().add_gradient(&DataContainer::one());
        loss_node.borrow_mut().apply_jacobian();

        self.decay_type.update_timestep(self.time_step);
    }

    fn create_config(&self) -> Config {
        let classifier_config = ClassifierConfig::from_network(self);
        Config::Classifier(classifier_config)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Array1};
    use rand::{distributions::Uniform, prelude::Distribution};

    use crate::{
        data::{data_container::DataContainer, Data},
        network::{types::classifier::ClassifierNetwork, Network},
        optimization::{learning_decay::LearningDecayType, momentum::DescentType},
        regularization::{dropout::NetworkMaskType, penalty::PenaltyConfig},
    };

    #[test]
    fn classification_test() {
        let penalty_config: PenaltyConfig = PenaltyConfig::none();

        let mut classifier: ClassifierNetwork = ClassifierNetwork::new(
            vec![1],
            vec![2],
            vec![2],
            penalty_config,
            NetworkMaskType::None,
            LearningDecayType::rms_prop(0.05, 0.95),
            DescentType::nesterov(0.95),
        );

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let before_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let before_output = classifier.predict(before_data);
        println!("Before: {:?}", before_output);

        let mut rng = rand::thread_rng();
        let low_distribution = Uniform::new(-1.0, -0.5);
        let high_distribution = Uniform::new(0.5, 1.0);

        let sample_distribution = Uniform::new(0.0, 1.0);

        for _i in 0..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 0..8 {
                let rand = sample_distribution.sample(&mut rng);
                if rand < 0.5 {
                    let x: f32 = low_distribution.sample(&mut rng);

                    inputs.push(Data::VectorF32(arr1(&[x])));
                    responses.push(Data::VectorF32(arr1(&[1.0, 0.0])));
                } else {
                    let x: f32 = high_distribution.sample(&mut rng);

                    inputs.push(Data::VectorF32(arr1(&[x])));
                    responses.push(Data::VectorF32(arr1(&[0.0, 1.0])));
                }
            }

            let input = DataContainer::Batch(inputs);
            let response = DataContainer::Batch(responses);

            classifier.train(input, response);
        }

        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("After: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[0.6]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("After 2: {:?}", after_output2);

        classifier
            .save_to_file("test/classifier_test.json")
            .expect("Save failed");
    }

    #[test]
    fn classifier_load_test() {
        let mut classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/classifier_test.json");

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[0.6]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);

        let mut inputs = Vec::new();
        let mut responses = Vec::new();

        let mut rng = rand::thread_rng();
        let low_distribution = Uniform::new(-1.0, -0.5);
        let high_distribution = Uniform::new(0.5, 1.0);

        let sample_distribution = Uniform::new(0.0, 1.0);

        for _j in 0..8 {
            let rand = sample_distribution.sample(&mut rng);
            if rand < 0.5 {
                let x: f32 = low_distribution.sample(&mut rng);

                inputs.push(Data::VectorF32(arr1(&[x])));
                responses.push(Data::VectorF32(arr1(&[1.0, 0.0])));
            } else {
                let x: f32 = high_distribution.sample(&mut rng);

                inputs.push(Data::VectorF32(arr1(&[x])));
                responses.push(Data::VectorF32(arr1(&[0.0, 1.0])));
            }
        }

        let input = DataContainer::Batch(inputs);
        let response = DataContainer::Batch(responses);

        classifier.train(input, response);
    }
}
