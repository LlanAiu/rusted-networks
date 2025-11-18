// builtin

// external
use serde::{Deserialize, Serialize};

use crate::{
    data::data_container::{ContainerType, DataContainer},
    optimization::{learning_decay::LearningRateParams, momentum::MomentumParams},
};

// internal

#[derive(Serialize, Deserialize)]
pub struct LayerParams {
    dim: Vec<usize>,
    parameters: Vec<f32>,
    momentum: MomentumParams,
    learning_rate: LearningRateParams,
}

impl LayerParams {
    pub fn null() -> LayerParams {
        LayerParams {
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
    ) -> LayerParams {
        LayerParams {
            dim,
            parameters,
            momentum,
            learning_rate,
        }
    }

    pub fn new_from_parameters(dim: Vec<usize>, parameters: Vec<f32>) -> LayerParams {
        LayerParams {
            dim,
            parameters,
            momentum: MomentumParams::null(),
            learning_rate: LearningRateParams::null(),
        }
    }

    pub fn get_parameters(&self) -> DataContainer {
        DataContainer::from_dim(&self.dim, self.parameters.clone(), ContainerType::Parameter)
    }

    pub fn get_momentum(&self) -> DataContainer {
        let momentum_vec: Vec<f32> = self.momentum.get_momentum();

        if momentum_vec.is_empty() {
            return DataContainer::Empty;
        }

        DataContainer::from_dim(&self.dim, momentum_vec, ContainerType::Parameter)
    }

    pub fn get_learning_rate(&self) -> DataContainer {
        let learning_vec: Vec<f32> = self.learning_rate.get_adaptive_learning_rate();

        if learning_vec.is_empty() {
            return DataContainer::Empty;
        }

        DataContainer::from_dim(&self.dim, learning_vec, ContainerType::Parameter)
    }
}
