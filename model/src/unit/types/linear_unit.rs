// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::config_types::{
        batch_norm_params::BatchNormParams, layer_params::LayerParams,
        learned_params::LearnedParams, unit_params::UnitParams,
    },
    node::NodeRef,
    optimization::{
        batch_norm::{BatchNormModule, NormalizationType},
        learning_decay::LearningDecayType,
        momentum::DescentType,
    },
    regularization::dropout::{NetworkMode, UnitMaskType},
    unit::{unit_base::UnitBase, Unit, UnitRef},
};
mod init;

pub struct LinearUnit<'a> {
    base: UnitBase<'a>,
    weights: NodeRef<'a>,
    biases: Option<NodeRef<'a>>,
    norm_module: Option<BatchNormModule<'a>>,
    input_size: usize,
    output_size: usize,
    activation: String,
    mask_type: UnitMaskType,
}

impl<'a> LinearUnit<'a> {
    pub fn from_config(
        config: &UnitParams,
        decay_type: LearningDecayType,
        descent_type: DescentType,
        normalization_type: NormalizationType,
    ) -> LinearUnit<'a> {
        init::build_linear_unit_from_config(config, decay_type, descent_type, normalization_type)
    }

    pub fn get_weights_params(&self) -> LayerParams {
        let weights_params = self.weights.borrow().save_parameters();
        if let LearnedParams::Layer { params } = weights_params {
            return params;
        }
        panic!("Got invalid LearnedParams format for layer weights!");
    }

    pub fn get_biases_params(&self) -> LayerParams {
        if let Option::Some(biases) = &self.biases {
            let biases_params = biases.borrow().save_parameters();
            if let LearnedParams::Layer { params } = biases_params {
                return params;
            }
            panic!("Got invalid LearnedParams format for layer biases!");
        }
        LayerParams::null()
    }

    pub fn get_batch_norm_params(&self) -> BatchNormParams {
        if let Option::Some(batch_norm) = &self.norm_module {
            return batch_norm.get_params();
        }

        BatchNormParams::null()
    }

    pub fn get_weights_ref(&self) -> &NodeRef<'a> {
        &self.weights
    }

    pub fn set_biases(&self, data: &LayerParams) {
        if let Option::Some(biases) = &self.biases {
            let parameters: DataContainer = data.get_parameters();
            let momentum: DataContainer = data.get_momentum();
            let learning_rate: DataContainer = data.get_learning_rate();

            biases.borrow_mut().set_data(parameters);
            if !matches!(&momentum, DataContainer::Empty) {
                biases.borrow_mut().set_momentum(momentum);
            }
            if !matches!(&learning_rate, DataContainer::Empty) {
                biases.borrow_mut().set_learning_rate(learning_rate);
            }
        }
    }

    pub fn set_weights(&self, data: &LayerParams) {
        let weights = data.get_parameters();
        let momentum = data.get_momentum();
        let learning_rate = data.get_learning_rate();

        self.weights.borrow_mut().set_data(weights);
        if !matches!(&momentum, DataContainer::Empty) {
            self.weights.borrow_mut().set_momentum(momentum);
        }
        if !matches!(&learning_rate, DataContainer::Empty) {
            self.weights.borrow_mut().set_learning_rate(learning_rate);
        }
    }

    pub fn set_normalization(&self, norm_params: &BatchNormParams) {
        if !norm_params.is_null() && !self.is_last_layer() {
            if let Option::Some(module) = &self.norm_module {
                module.set_parameters(norm_params);
                return;
            }
            println!("Detected BatchNormParams wasn't null but couldn't find BatchNormModule -- skipping assignment");
        }
    }

    pub fn get_input_size(&self) -> usize {
        self.input_size
    }

    pub fn get_output_size(&self) -> usize {
        self.output_size
    }

    pub fn get_weights(&self) -> Vec<f32> {
        let data: DataContainer = self.weights.borrow_mut().get_data();

        if let DataContainer::Parameter(Data::MatrixF32(matrix)) = data {
            return matrix.flatten().to_vec();
        }

        println!("Couldn't package weights for serialization due to invalid data container/type");
        Vec::new()
    }

    pub fn get_biases(&self) -> Vec<f32> {
        if let Option::Some(biases) = &self.biases {
            let data = biases.borrow_mut().get_data();
            if let DataContainer::Parameter(Data::VectorF32(vec)) = data {
                return vec.to_vec();
            }

            println!(
                "Couldn't package biases for serialization due to invalid data container/type"
            );
        }
        Vec::new()
    }

    pub fn get_activation(&self) -> &str {
        &self.activation
    }

    pub fn get_mask_type(&self) -> &UnitMaskType {
        &self.mask_type
    }

    pub fn is_last_layer(&self) -> bool {
        self.base.is_last_layer()
    }
}

impl<'a> Unit<'a> for LinearUnit<'a> {
    fn add_input(&mut self, this: &UnitRef<'a>, input: &UnitRef<'a>) {
        self.base.add_input(this, input);
    }

    fn add_output(&mut self, output: &UnitRef<'a>) {
        self.base.add_output(output);
    }

    fn get_inputs(&self) -> &Vec<UnitRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<UnitRef<'a>> {
        self.base.get_outputs()
    }

    fn get_output_node(&self) -> &NodeRef<'a> {
        self.base.get_output_node()
    }

    fn update_mode(&mut self, new_mode: NetworkMode) {
        self.base.update_mode(new_mode);

        for unit in self.base.get_outputs() {
            unit.borrow_mut().update_mode(new_mode);
        }
    }
}
