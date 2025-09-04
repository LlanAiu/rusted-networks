// builtin

use core::panic;

use ndarray::{arr1, Array2};
// external
use serde::{Deserialize, Serialize};

use crate::{
    data::{data_container::DataContainer, Data},
    optimization::{learning_decay::LearningRateParams, momentum::MomentumParams},
};

// internal

#[derive(Serialize, Deserialize)]
pub struct LearnedParams {
    dim: Vec<usize>,
    parameters: Vec<f32>,
    momentum: MomentumParams,
    learning_rate: LearningRateParams,
}

impl LearnedParams {
    pub fn null() -> LearnedParams {
        LearnedParams {
            dim: Vec::new(),
            parameters: Vec::new(),
            momentum: MomentumParams::null(),
            learning_rate: LearningRateParams::null(),
        }
    }

    pub fn new(
        dim: Vec<usize>,
        parameters: Vec<f32>,
        momentum: MomentumParams,
        learning_rate: LearningRateParams,
    ) -> LearnedParams {
        LearnedParams {
            dim,
            parameters,
            momentum,
            learning_rate,
        }
    }

    pub fn get_parameters(&self) -> DataContainer {
        if self.dim.len() == 0 {
            if self.parameters.len() > 0 {
                return DataContainer::Parameter(Data::ScalarF32(self.parameters[0]));
            }
            println!("DataContainer::Empty returned on null data");
            return DataContainer::Empty;
        } else if self.dim.len() == 1 {
            if self.parameters.len() == self.dim[0] {
                return DataContainer::Parameter(Data::VectorF32(arr1(&self.parameters)));
            }
            println!("DataContainer::Empty returned on mismatched dimensions and data");
            return DataContainer::Empty;
        } else if self.dim.len() == 2 {
            if self.parameters.len() == self.dim[0] * self.dim[1] {
                let matrix =
                    Array2::from_shape_vec((self.dim[0], self.dim[1]), self.parameters.clone())
                        .expect("Couldn't create matrix from shape");

                return DataContainer::Parameter(Data::MatrixF32(matrix));
            }
            println!("DataContainer::Empty returned on mismatched dimensions and data");
            return DataContainer::Empty;
        } else {
            panic!("Unsupported dimensions to coerce to data type!");
        }
    }

    pub fn new_from_parameters(dim: Vec<usize>, parameters: Vec<f32>) -> LearnedParams {
        LearnedParams {
            dim,
            parameters,
            momentum: MomentumParams::null(),
            learning_rate: LearningRateParams::null(),
        }
    }

    pub fn get_momentum(&self) -> DataContainer {
        todo!()
    }

    pub fn get_learning_rate(&self) -> DataContainer {
        todo!()
    }
}
