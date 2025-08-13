// builtin
use std::{
    fs::{read_to_string, write},
    io::{Error, ErrorKind, Result},
};

// external
use serde::{Deserialize, Serialize};

// internal
use crate::{
    network::{
        config_types::{
            hyper_params::HyperParams, input_params::InputParams, loss_params::LossParams,
            regularization_params::RegularizationParams, unit_params::UnitParams, Config,
        },
        types::regressor::RegressorNetwork,
    },
    regularization::penalty::PenaltyConfig,
};

#[derive(Serialize, Deserialize)]
pub struct RegressorConfig {
    input: InputParams,
    units: Vec<UnitParams>,
    loss: LossParams,
    hyperparams: HyperParams,
    regularization: RegularizationParams,
}

impl RegressorConfig {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        learning_rate: f32,
        penalty_config: PenaltyConfig,
        with_dropout: bool,
    ) -> RegressorConfig {
        if input_size.len() != 1 || output_size.len() != 1 {
            panic!("[SIMPLE_REGRESSOR] Invalid input / output dimensions for network type, expected 1 and 1 but got {} and {}.", input_size.len(), output_size.len());
        }
        let input_usize = input_size[0];
        let output_usize = output_size[0];

        let input: InputParams = InputParams {
            input_size: input_size,
        };
        let loss: LossParams = LossParams {
            loss_type: String::from("mean_squared_error"),
            output_size: output_size,
        };
        let hyperparams: HyperParams = HyperParams::new(learning_rate);

        let mut units: Vec<UnitParams> = Vec::new();
        let mut prev_width: usize = input_usize;

        for unit_size in hidden_sizes {
            let unit: UnitParams = UnitParams::new_linear(prev_width, unit_size, "relu");
            units.push(unit);
            prev_width = unit_size;
        }

        let inference_unit: UnitParams = UnitParams::new_linear(prev_width, output_usize, "none");
        units.push(inference_unit);

        let regularization: RegularizationParams =
            RegularizationParams::from_builder(penalty_config.get_builder(), with_dropout);

        RegressorConfig {
            input,
            units,
            loss,
            hyperparams,
            regularization,
        }
    }

    pub fn to_config(network: &RegressorNetwork) -> RegressorConfig {
        let input = InputParams::from_unit(&network.input);

        let loss = LossParams::from_unit(&network.loss);

        let mut units: Vec<UnitParams> = Vec::new();

        for unit in &network.hidden {
            units.push(UnitParams::from_linear_unit(unit));
        }
        units.push(UnitParams::from_linear_unit(&network.inference));

        let hyperparams: HyperParams = HyperParams::new(network.learning_rate);

        let regularization: RegularizationParams =
            RegularizationParams::new(network.penalty_type.clone(), network.with_dropout);

        RegressorConfig {
            input,
            units,
            loss,
            hyperparams,
            regularization,
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

    pub fn params(&self) -> &HyperParams {
        &self.hyperparams
    }

    pub fn regularization(&self) -> &RegularizationParams {
        &self.regularization
    }
}
