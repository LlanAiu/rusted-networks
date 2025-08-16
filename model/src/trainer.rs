// builtin

// external
use rand::random_bool;

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::{config_types::Config, Network},
    trainer::{error::PredictionError, examples::SupervisedExample},
};
pub mod error;
pub mod examples;

pub struct SupervisedTrainer<N, T>
where
    N: Network,
    T: SupervisedExample,
{
    model: N,
    train: Vec<T>,
    test: Vec<T>,
    //TODO: Stopping criteria
}

impl<N, T> SupervisedTrainer<N, T>
where
    N: Network,
    T: SupervisedExample,
{
    pub fn new(network: N, train: Vec<T>, test: Vec<T>) -> SupervisedTrainer<N, T> {
        SupervisedTrainer {
            model: network,
            train,
            test,
        }
    }

    pub fn new_with_split(network: N, data: Vec<T>) -> SupervisedTrainer<N, T> {
        let mut train: Vec<T> = Vec::new();
        let mut test: Vec<T> = Vec::new();

        for example in data {
            let is_train = random_bool(0.8);

            if is_train {
                train.push(example);
            } else {
                test.push(example);
            }
        }

        SupervisedTrainer {
            model: network,
            train,
            test,
        }
    }

    pub fn train_epoch(&self) -> (Config, PredictionError) {
        let mut inputs: Vec<Data> = Vec::new();
        let mut responses: Vec<Data> = Vec::new();

        for example in self.train.iter() {
            inputs.push(example.get_input());
            responses.push(example.get_response());

            if inputs.len() == 8 {
                self.model.train(
                    DataContainer::Batch(inputs.clone()),
                    DataContainer::Batch(responses.clone()),
                );

                inputs.clear();
                responses.clear();
            }
        }

        let mut error_sum: PredictionError = PredictionError::empty();
        for example in self.test.iter() {
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

        for i in 1..21 {
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
        trainer::{examples::QuadraticExample, SupervisedTrainer},
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
            RegressorNetwork::new(vec![1], vec![1], vec![6, 3], 0.005, config, false);

        let trainer: SupervisedTrainer<RegressorNetwork, QuadraticExample> =
            SupervisedTrainer::new(regressor, train, test);

        trainer.train("test/quadratic_training.json");
    }
}
