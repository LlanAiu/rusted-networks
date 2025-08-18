// builtin
use std::mem::take;

// external

// internal
use crate::trainer::{
    data_subsets::{DataSplitter, DataSubsets},
    examples::SupervisedExample,
};

pub struct TrainerConfig<T: SupervisedExample> {
    total_iterations: usize,
    batch_size: usize,
    train: Vec<T>,
    test: Vec<T>,
}

impl<T> TrainerConfig<T>
where
    T: SupervisedExample,
{
    pub fn new(
        total_iterations: usize,
        batch_size: usize,
        train: Vec<T>,
        test: Vec<T>,
    ) -> TrainerConfig<T> {
        TrainerConfig {
            total_iterations,
            batch_size,
            train,
            test,
        }
    }

    pub fn new_with_split(
        total_iterations: usize,
        batch_size: usize,
        data: Vec<T>,
        splitter: impl DataSplitter,
    ) -> TrainerConfig<T> {
        let mut subsets: DataSubsets<T> = splitter.split(data);

        TrainerConfig {
            total_iterations,
            batch_size,
            train: subsets.take_train(),
            test: subsets.take_test(),
        }
    }

    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn train_ref(&self) -> &Vec<T> {
        &self.train
    }

    pub fn test_ref(&self) -> &Vec<T> {
        &self.test
    }

    pub fn take_train(&mut self) -> Vec<T> {
        take(&mut self.train)
    }

    pub fn take_test(&mut self) -> Vec<T> {
        take(&mut self.test)
    }
}
