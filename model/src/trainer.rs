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
            let error = example.get_error(predicted);

            error_sum = error_sum.plus(&error);
        }

        (Config::None, error_sum)
    }

    pub fn train(&self, save_path: &str) {
        let (config, error) = self.train_epoch();

        let mut prev_config: Config = config;
        let mut prev_error: PredictionError = error;

        for _i in 0..5 {
            let (config, error) = self.train_epoch();

            if prev_error < error {
                prev_config.save_to_file(save_path).expect("Save Failed");
                break;
            }

            prev_config = config;
            prev_error = error;
        }
    }
}
