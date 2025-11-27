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
    optimization::{
        batch_norm::NormalizationType, learning_decay::LearningDecayType, momentum::DescentType,
    },
    regularization::{
        dropout::{NetworkMaskType, UnitMaskType},
        penalty::PenaltyConfig,
    },
};

#[derive(Serialize, Deserialize)]
pub struct RegressorConfig {
    input: InputParams,
    units: Vec<UnitParams>,
    loss: LossParams,
    hyperparams: HyperParams,
    regularization: RegularizationParams,
    time_step: usize,
}

impl RegressorConfig {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        penalty_config: PenaltyConfig,
        mask_type: NetworkMaskType,
        decay_type: LearningDecayType,
        descent_type: DescentType,
        normalization_type: NormalizationType,
    ) -> RegressorConfig {
        if input_size.len() != 1 || output_size.len() != 1 {
            panic!("[SIMPLE_REGRESSOR] Invalid input / output dimensions for network type, expected 1 and 1 but got {} and {}.", input_size.len(), output_size.len());
        }
        let input_usize = input_size[0];
        let output_usize = output_size[0];

        let input: InputParams = InputParams::new(input_size, mask_type.input_probability());
        let loss: LossParams = LossParams {
            loss_type: String::from("mean_squared_error"),
            output_size: output_size,
        };
        let hyperparams: HyperParams =
            HyperParams::new(decay_type, descent_type, normalization_type.clone());

        let mut units: Vec<UnitParams> = Vec::new();
        let mut prev_width: usize = input_usize;

        let hidden_keep_p = mask_type.hidden_probability();
        for unit_size in hidden_sizes {
            let unit: UnitParams = UnitParams::new_linear(
                prev_width,
                unit_size,
                "relu",
                UnitMaskType::from_keep_probability(hidden_keep_p),
                normalization_type.clone(),
                false,
            );
            units.push(unit);
            prev_width = unit_size;
        }

        let inference_unit: UnitParams = UnitParams::new_linear(
            prev_width,
            output_usize,
            "none",
            UnitMaskType::from_keep_probability(hidden_keep_p),
            normalization_type,
            true,
        );
        units.push(inference_unit);

        let regularization: RegularizationParams =
            RegularizationParams::from_builder(penalty_config.get_builder());

        RegressorConfig {
            input,
            units,
            loss,
            hyperparams,
            regularization,
            time_step: 0,
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

        let hyperparams: HyperParams = HyperParams::new(
            network.decay_type.clone(),
            network.descent_type.clone(),
            network.normalization_type.clone(),
        );

        let regularization: RegularizationParams =
            RegularizationParams::new(network.penalty_type.clone());

        RegressorConfig {
            input,
            units,
            loss,
            hyperparams,
            regularization,
            time_step: network.time_step,
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

    pub fn timestep(&self) -> usize {
        self.time_step
    }
}
