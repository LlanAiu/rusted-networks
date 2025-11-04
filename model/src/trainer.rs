// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::{config_types::Config, Network},
    trainer::{error::PredictionError, examples::SupervisedExample, trainer_params::TrainerConfig},
};
pub mod data_subsets;
pub mod error;
pub mod examples;
pub mod trainer_params;

pub struct SupervisedTrainer<N, T>
where
    N: Network,
    T: SupervisedExample,
{
    model: N,
    config: TrainerConfig<T>,
}

impl<N, T> SupervisedTrainer<N, T>
where
    N: Network,
    T: SupervisedExample,
{
    pub fn new(network: N, config: TrainerConfig<T>) -> SupervisedTrainer<N, T> {
        SupervisedTrainer {
            model: network,
            config,
        }
    }

    fn train_epoch(&mut self) -> (Config, PredictionError) {
        let mut inputs: Vec<Data> = Vec::new();
        let mut responses: Vec<Data> = Vec::new();

        for example in self.config.train_ref().iter() {
            inputs.push(example.get_input());
            responses.push(example.get_response());

            if inputs.len() == self.config.batch_size() {
                self.model.train(
                    DataContainer::Batch(inputs.clone()),
                    DataContainer::Batch(responses.clone()),
                );

                inputs.clear();
                responses.clear();
            }
        }

        if inputs.len() != 0 {
            self.model.train(
                DataContainer::Batch(inputs.clone()),
                DataContainer::Batch(responses.clone()),
            );

            inputs.clear();
            responses.clear();
        }

        let mut error_sum: PredictionError = PredictionError::empty();
        for example in self.config.test_ref().iter() {
            let input = DataContainer::Inference(example.get_input());
            let predicted = self.model.predict(input);
            let error = example.get_test_error(predicted);

            if error_sum.is_empty() {
                error_sum = error_sum.plus(&error);
            } else {
                error_sum.plus_assign(&error);
            }
        }

        (Config::from_network(&self.model), error_sum)
    }

    pub fn train(&mut self, save_path: &str) {
        let (config, error) = self.train_epoch();

        println!("Test error 0: {:?}\n", error);

        let mut prev_config: Config = config;
        let mut prev_error: PredictionError = error;

        for i in 1..self.config.total_iterations() {
            let (config, error) = self.train_epoch();

            println!("Test error {i}: {:?}\n", error);

            if prev_error < error {
                prev_config.save_to_file(save_path).expect("Save Failed");
                println!("Training stopped after {i} iterations");
                return;
            }

            prev_config = config;
            prev_error = error;
        }

        prev_config.save_to_file(save_path).expect("Save Failed");
        println!("Training finished.");
    }

    pub fn evaluate(&self) -> PredictionError {
        let mut error_sum: PredictionError = PredictionError::empty();
        for example in self.config.test_ref().iter() {
            let input = DataContainer::Inference(example.get_input());
            let predicted = self.model.predict(input);
            let error = example.get_test_error(predicted);

            if error_sum.is_empty() {
                error_sum = error_sum.plus(&error);
            } else {
                error_sum.plus_assign(&error);
            }
        }

        error_sum
    }
}

#[cfg(test)]

mod tests {

    use rand::{distributions::Uniform, prelude::Distribution};

    use crate::{
        network::types::regressor::RegressorNetwork,
        optimization::{learning_decay::LearningDecayType, momentum::DescentType},
        regularization::penalty::{l2_penalty::builder::L2PenaltyBuilder, PenaltyConfig},
        trainer::{examples::QuadraticExample, trainer_params::TrainerConfig, SupervisedTrainer},
    };

    #[test]
    fn trainer_test() {
        let mut train: Vec<QuadraticExample> = Vec::new();
        let mut test: Vec<QuadraticExample> = Vec::new();
        let mut rng = rand::thread_rng();
        let distribution = Uniform::new(1.0, 4.0);

        for _i in 0..1600 {
            let x: f32 = distribution.sample(&mut rng);
            let example: QuadraticExample = QuadraticExample::new(x);
            train.push(example);
        }

        for i in 0..16 {
            let x: f32 = (i as f32) / 5.0 + 1.0;
            let example: QuadraticExample = QuadraticExample::new(x);
            test.push(example);
        }

        let config: PenaltyConfig = PenaltyConfig::new(L2PenaltyBuilder::new(0.2));
        let regressor: RegressorNetwork = RegressorNetwork::new(
            vec![1],
            vec![1],
            vec![12, 6],
            config,
            false,
            LearningDecayType::constant(0.001),
            DescentType::nesterov(0.95),
        );

        let train_config: TrainerConfig<QuadraticExample> = TrainerConfig::new(25, 16, train, test);

        let mut trainer: SupervisedTrainer<RegressorNetwork, QuadraticExample> =
            SupervisedTrainer::new(regressor, train_config);

        trainer.train("test/quadratic_training.json");
    }
}
