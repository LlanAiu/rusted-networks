// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::network::types::{
    classifier::config::ClassifierConfig, regressor::config::RegressorConfig,
};
pub mod hyper_params;
pub mod input_params;
pub mod loss_params;
pub mod regularization_params;
pub mod unit_params;

#[derive(Serialize, Deserialize)]
pub enum Config {
    Classifier(ClassifierConfig),
    Regressor(RegressorConfig),
}
