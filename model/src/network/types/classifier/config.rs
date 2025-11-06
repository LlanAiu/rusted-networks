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
        types::classifier::ClassifierNetwork,
    },
    optimization::{learning_decay::LearningDecayType, momentum::DescentType},
    regularization::penalty::PenaltyConfig,
};

#[derive(Serialize, Deserialize)]
pub struct ClassifierConfig {
    input: InputParams,
    units: Vec<UnitParams>,
    loss: LossParams,
    params: HyperParams,
    regularization: RegularizationParams,
    time_step: usize,
}

impl ClassifierConfig {
    pub fn new(
        input_size: Vec<usize>,
        output_size: Vec<usize>,
        hidden_sizes: Vec<usize>,
        penalty_config: PenaltyConfig,
        with_dropout: bool,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> ClassifierConfig {
        if input_size.len() != 1 || output_size.len() != 1 {
            panic!("[SIMPLE_CLASSIFIER] Invalid input / output dimensions for network type, expected 1 and 1 but got {} and {}.", input_size.len(), output_size.len());
        }
        let input_usize: usize = input_size[0];
        let output_usize: usize = output_size[0];

        let input: InputParams = InputParams { input_size };
        let loss: LossParams = LossParams {
            loss_type: String::from("base_cross_entropy"),
            output_size,
        };

        let params: HyperParams = HyperParams::new(decay_type, descent_type);

        let mut units: Vec<UnitParams> = Vec::new();
        let mut prev_width: usize = input_usize;

        for unit_size in hidden_sizes {
            let unit: UnitParams = UnitParams::new_linear(prev_width, unit_size, "relu", false);
            units.push(unit);
            prev_width = unit_size;
        }

        let inference_unit: UnitParams =
            UnitParams::new_softmax(prev_width, output_usize, "none", true);
        units.push(inference_unit);

        let regularization: RegularizationParams =
            RegularizationParams::from_builder(penalty_config.get_builder(), with_dropout);

        ClassifierConfig {
            input,
            units,
            loss,
            params,
            regularization,
            time_step: 0,
        }
    }

    pub fn from_network(network: &ClassifierNetwork) -> ClassifierConfig {
        let input: InputParams = InputParams::from_unit(&network.input);

        let loss: LossParams = LossParams::from_unit(&network.loss);

        let mut units: Vec<UnitParams> = Vec::new();

        for unit in &network.hidden {
            units.push(UnitParams::from_linear_unit(unit));
        }
        units.push(UnitParams::from_softmax_unit(&network.inference));

        let params: HyperParams =
            HyperParams::new(network.decay_type.clone(), network.descent_type.clone());

        let regularization: RegularizationParams =
            RegularizationParams::new(network.penalty_type.clone(), network.with_dropout);

        ClassifierConfig {
            input,
            units,
            loss,
            params,
            regularization,
            time_step: network.time_step,
        }
    }

    pub fn save_to_file(self, path: &str) -> Result<()> {
        let config: Config = Config::Classifier(self);
        let json_string = serde_json::to_string_pretty(&config).unwrap();
        write(path, json_string)
    }

    pub fn load_from_file(path: &str) -> Result<ClassifierConfig> {
        let data = read_to_string(path)?;
        let config: Config =
            serde_json::from_str(&data).expect("Invalid JSON data for network configuration");

        if let Config::Classifier(class_config) = config {
            return Ok(class_config);
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
        &self.params
    }

    pub fn regularization(&self) -> &RegularizationParams {
        &self.regularization
    }

    pub fn timestep(&self) -> usize {
        self.time_step
    }
}
