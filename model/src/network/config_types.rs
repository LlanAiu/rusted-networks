// builtin
use std::io::Result;

// external
use serde::{Deserialize, Serialize};

// internal
use crate::network::{
    types::{classifier::config::ClassifierConfig, regressor::config::RegressorConfig},
    Network,
};
pub mod hyper_params;
pub mod input_params;
pub mod learned_params;
pub mod loss_params;
pub mod regularization_params;
pub mod unit_params;

#[derive(Serialize, Deserialize)]
pub enum Config {
    Classifier(ClassifierConfig),
    Regressor(RegressorConfig),
    None,
}

impl Config {
    pub fn save_to_file(self, path: &str) -> Result<()> {
        match self {
            Config::Classifier(classifier_config) => classifier_config.save_to_file(path),
            Config::Regressor(regressor_config) => regressor_config.save_to_file(path),
            Config::None => Ok(()),
        }
    }

    pub fn from_network(network: &impl Network) -> Config {
        network.create_config()
    }
}
