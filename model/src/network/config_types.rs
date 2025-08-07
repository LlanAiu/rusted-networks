// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::network::types::{
    binary_classifier::config::BinaryClassifierConfig, simple_classifier::config::ClassifierConfig,
    simple_regressor::config::RegressorConfig,
};
pub mod hyper_params;
pub mod input_params;
pub mod loss_params;
pub mod unit_params;

#[derive(Serialize, Deserialize)]
pub enum Config {
    BinaryClassifier(BinaryClassifierConfig),
    Classifier(ClassifierConfig),
    Regressor(RegressorConfig),
}
