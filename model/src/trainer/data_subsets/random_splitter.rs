// builtin

// external

// internal

use ndarray_rand::rand_distr::num_traits::clamp;
use rand::random_bool;

use crate::trainer::data_subsets::{DataSplitter, DataSubsets};

pub struct RandomSplitter {
    split_prob: f64,
}

impl RandomSplitter {
    pub fn new(split_prob: f64) -> RandomSplitter {
        RandomSplitter {
            split_prob: clamp(split_prob, 0.0, 1.0),
        }
    }
}

impl DataSplitter for RandomSplitter {
    fn split<T>(&self, data: Vec<T>) -> DataSubsets<T> {
        let mut train: Vec<T> = Vec::new();
        let mut test: Vec<T> = Vec::new();

        for example in data {
            let is_train = random_bool(self.split_prob);

            if is_train {
                train.push(example);
            } else {
                test.push(example);
            }
        }

        DataSubsets::new(train, test)
    }
}
