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

    pub fn train_epoch(&self) -> (Config, PredictionError) {
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

        let mut error_sum: PredictionError = PredictionError::empty();
        for example in self.config.test_ref().iter() {
            let input = DataContainer::Inference(example.get_input());
            let predicted = self.model.predict(input);
            let error = example.get_test_error(predicted);

            error_sum = error_sum.plus(&error);
        }

        (Config::from_network(&self.model), error_sum)
    }

    pub fn train(&self, save_path: &str) {
        let (config, error) = self.train_epoch();

        let mut prev_config: Config = config;
        let mut prev_error: PredictionError = error;

        for i in 0..self.config.total_iterations() {
            let (config, error) = self.train_epoch();

            println!("Test error {i}: {:?}", error);

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
}

#[cfg(test)]

mod tests {
    use rand::random_range;

    use crate::{
        network::types::regressor::RegressorNetwork,
        regularization::penalty::{l2_penalty::builder::L2PenaltyBuilder, PenaltyConfig},
        trainer::{examples::QuadraticExample, trainer_params::TrainerConfig, SupervisedTrainer},
    };

    #[test]
    fn trainer_test() {
        let mut train: Vec<QuadraticExample> = Vec::new();
        let mut test: Vec<QuadraticExample> = Vec::new();

        for _i in 0..1600 {
            let x: f32 = random_range(1.0..4.0);
            let example: QuadraticExample = QuadraticExample::new(x);
            train.push(example);
        }

        for i in 0..16 {
            let x: f32 = (i as f32) / 5.0 + 1.0;
            let example: QuadraticExample = QuadraticExample::new(x);
            test.push(example);
        }

        let config: PenaltyConfig = PenaltyConfig::new(L2PenaltyBuilder::new(0.2));
        let regressor: RegressorNetwork =
            RegressorNetwork::new(vec![1], vec![1], vec![12, 6], 0.001, config, false);

        let train_config: TrainerConfig<QuadraticExample> = TrainerConfig::new(25, 16, train, test);

        let trainer: SupervisedTrainer<RegressorNetwork, QuadraticExample> =
            SupervisedTrainer::new(regressor, train_config);

        trainer.train("test/quadratic_training.json");
    }
}
