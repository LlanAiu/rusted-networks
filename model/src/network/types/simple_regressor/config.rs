// builtin
use std::{
    fs::{read_to_string, write},
    io::{Error, ErrorKind, Result},
};

// external
use serde::{Deserialize, Serialize};

// internal
use crate::network::config_types::{Config, InputParams, LearningParams, LossParams, UnitParams};

#[derive(Serialize, Deserialize)]
pub struct RegressorConfig {
    input: InputParams,
    units: Vec<UnitParams>,
    loss: LossParams,
    learning: LearningParams,
}

impl RegressorConfig {
    pub fn new(
        input: InputParams,
        units: Vec<UnitParams>,
        loss: LossParams,
        learning: LearningParams,
    ) -> RegressorConfig {
        RegressorConfig {
            input,
            units,
            loss,
            learning,
        }
    }

    pub fn save_to_file(self, path: &str) -> Result<()> {
        let config: Config = Config::Regressor(self);
        let json_string = serde_json::to_string_pretty(&config).unwrap();
        write(path, json_string)
    }

    pub fn load_from_file(path: &str) -> Result<RegressorConfig> {
        let data = read_to_string(path)?;
        let config: Config =
            serde_json::from_str(&data).expect("Invalid JSON data for network configuration");

        if let Config::Regressor(regression_config) = config {
            return Ok(regression_config);
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
