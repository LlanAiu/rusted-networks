// builtin
use std::mem::take;

// external

// internal
pub mod random_splitter;

pub struct DataSubsets<T> {
    train: Vec<T>,
    test: Vec<T>,
}

impl<T> DataSubsets<T> {
    pub fn new(train: Vec<T>, test: Vec<T>) -> DataSubsets<T> {
        DataSubsets { train, test }
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

pub trait DataSplitter {
    fn split<T>(&self, data: Vec<T>) -> DataSubsets<T>;
}
