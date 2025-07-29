// builtin
use std::{
    fs::{read_to_string, write},
    io::{Error, ErrorKind, Result},
};

// external
use serde::{Deserialize, Serialize};

// internal
use crate::network::config_types::{
    input_params::InputParams, learning_params::LearningParams, loss_params::LossParams,
    unit_params::UnitParams, Config,
};

#[derive(Serialize, Deserialize)]
pub struct BinaryClassifierConfig {
    input: InputParams,
    units: Vec<UnitParams>,
    loss: LossParams,
    learning: LearningParams,
}

impl BinaryClassifierConfig {
    pub fn new(
        input: InputParams,
        units: Vec<UnitParams>,
        loss: LossParams,
        learning: LearningParams,
    ) -> BinaryClassifierConfig {
        BinaryClassifierConfig {
            input,
            units,
            loss,
            learning,
        }
    }

    pub fn save_to_file(self, path: &str) -> Result<()> {
        let config: Config = Config::BinaryClassifier(self);
        let json_string = serde_json::to_string_pretty(&config).unwrap();
        write(path, json_string)
    }

    pub fn load_from_file(path: &str) -> Result<BinaryClassifierConfig> {
        let data = read_to_string(path)?;
        let config: Config =
            serde_json::from_str(&data).expect("Invalid JSON data for network configuration");

        if let Config::BinaryClassifier(binary_config) = config {
            return Ok(binary_config);
        }

        Err(Error::new(
            ErrorKind::InvalidData,
            "JSON network data did not match the requested network type",
        ))
    }

    pub fn input(&self) -> &InputParams {
        &self.input
    }

    pub fn units(&self) -> &Vec<UnitParams> {
        &self.units
    }

    pub fn loss(&self) -> &LossParams {
        &self.loss
    }

    pub fn learning(&self) -> &LearningParams {
        &self.learning
    }
}
