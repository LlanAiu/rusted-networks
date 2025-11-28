// builtin
use core::panic;
use std::usize;

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::config_types::{
        batch_norm_params::{BatchNormParams, NormParams},
        layer_params::LayerParams,
        learned_params::LearnedParams,
        unit_params::UnitParams,
    },
    node::{
        types::{
            activation_node::ActivationNode, add_node::AddNode, bias_node::BiasNode,
            mask_node::MaskNode, matrix_multiply_node::MatrixMultiplyNode,
            multiply_node::MultiplyNode, normalization_node::NormalizationNode,
            weight_node::WeightNode,
        },
        NodeRef,
    },
    optimization::{
        batch_norm::{BatchNormModule, NormalizationType},
        learning_decay::LearningDecayType,
        momentum::DescentType,
    },
    regularization::dropout::{NetworkMode, UnitMaskType},
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct LinearUnit<'a> {
    base: UnitBase<'a>,
    weights: NodeRef<'a>,
    biases: NodeRef<'a>,
    norm_module: Option<BatchNormModule<'a>>,
    input_size: usize,
    output_size: usize,
    activation: String,
    mask_type: UnitMaskType,
}

impl<'a> LinearUnit<'a> {
    fn new(
        function: &str,
        input_size: usize,
        output_size: usize,
        decay_type: LearningDecayType,
        descent_type: DescentType,
        mask_type: UnitMaskType,
        normalization_type: NormalizationType,
        norm_params: &NormParams,
        is_last_layer: bool,
    ) -> LinearUnit<'a> {
        let weights_ref: NodeRef = NodeRef::new(WeightNode::new_matrix(
            input_size,
            output_size,
            decay_type.clone(),
            descent_type.clone(),
        ));
        let biases_ref: NodeRef = NodeRef::new(BiasNode::new(
            output_size,
            decay_type.clone(),
            descent_type.clone(),
        ));
        let matmul_ref: NodeRef = NodeRef::new(MatrixMultiplyNode::new());
        let add_ref: NodeRef = NodeRef::new(AddNode::new());
        let activation_ref: NodeRef = NodeRef::new(ActivationNode::new(function));

        matmul_ref.borrow_mut().add_input(&matmul_ref, &weights_ref);

        add_ref.borrow_mut().add_input(&add_ref, &matmul_ref);
        add_ref.borrow_mut().add_input(&add_ref, &biases_ref);

        activation_ref
            .borrow_mut()
            .add_input(&activation_ref, &add_ref);

        let mut output_ref: &NodeRef = &activation_ref;
        let mut norm_module: Option<BatchNormModule> = Option::None;
        let mut norm: Option<&NodeRef> = Option::None;

        let norm_add_ref: NodeRef;
        let norm_ref: NodeRef;

        if let NormalizationType::BatchNorm { decay } = &normalization_type {
            if norm_params.is_null() {
                norm_ref = NodeRef::new(NormalizationNode::new(*decay));
            } else {
                let mean = norm_params.get_mean();
                let variance = norm_params.get_variance();
                let decay = norm_params.get_decay();
                norm_ref = NodeRef::new(NormalizationNode::from_parameters(mean, variance, decay));
            }

            let scale_ref = NodeRef::new(WeightNode::new_vec(
                output_size,
                decay_type.clone(),
                descent_type.clone(),
            ));
            let shift_ref = NodeRef::new(BiasNode::new(output_size, decay_type, descent_type));
            let norm_multiply_ref: NodeRef = NodeRef::new(MultiplyNode::new());
            norm_add_ref = NodeRef::new(AddNode::new());

            norm_ref.borrow_mut().add_input(&norm_ref, &activation_ref);

            norm_multiply_ref
                .borrow_mut()
                .add_input(&norm_multiply_ref, &norm_ref);
            norm_multiply_ref
                .borrow_mut()
                .add_input(&norm_multiply_ref, &scale_ref);

            norm_add_ref
                .borrow_mut()
                .add_input(&norm_add_ref, &norm_multiply_ref);
            norm_add_ref
                .borrow_mut()
                .add_input(&norm_add_ref, &shift_ref);

            output_ref = &norm_add_ref;

            let module = BatchNormModule::new(&norm_ref, &scale_ref, &shift_ref);
            norm_module = Option::Some(module);
            norm = Option::Some(&norm_ref);
        }

        let mut mask: Option<&NodeRef> = Option::None;

        let mask_ref: NodeRef;
        let multiply_ref: NodeRef;

        if !is_last_layer {
            if let UnitMaskType::Dropout {
                keep_probability: probability,
            } = &mask_type
            {
                mask_ref = NodeRef::new(MaskNode::new(vec![output_size], *probability));
                multiply_ref = NodeRef::new(MultiplyNode::new());

                multiply_ref
                    .borrow_mut()
                    .add_input(&multiply_ref, &output_ref);

                multiply_ref
                    .borrow_mut()
                    .add_input(&multiply_ref, &mask_ref);

                mask = Option::Some(&mask_ref);
                output_ref = &multiply_ref
            }
        }

        LinearUnit {
            base: UnitBase::new(&matmul_ref, output_ref, mask, norm, is_last_layer),
            weights: weights_ref,
            biases: biases_ref,
            input_size,
            output_size,
            activation: function.to_string(),
            norm_module,
            mask_type,
        }
    }

    pub fn from_config(
        config: &UnitParams,
        decay_type: LearningDecayType,
        descent_type: DescentType,
        normalization_type: NormalizationType,
    ) -> LinearUnit<'a> {
        if let UnitParams::Linear {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            keep_probability,
            is_last_layer,
            norm_params,
        } = config
        {
            let unit: LinearUnit = Self::new(
                activation,
                *input_size,
                *output_size,
                decay_type,
                descent_type,
                UnitMaskType::from_keep_probability(*keep_probability),
                normalization_type,
                norm_params.get_normalization(),
                *is_last_layer,
            );

            unit.set_weights(weights);
            unit.set_biases(biases);

            unit.set_normalization(norm_params);

            return unit;
        }

        panic!("Mismatched unit parameter types for initialization: expected UnitParams::Linear but got {},", config.type_name());
    }

    pub fn get_weights_params(&self) -> LayerParams {
        let weights_params = self.weights.borrow().save_parameters();
        if let LearnedParams::Layer { params } = weights_params {
            return params;
        }
        panic!("Got invalid LearnedParams format for layer weights!");
    }

    pub fn get_biases_params(&self) -> LayerParams {
        let biases_params = self.biases.borrow().save_parameters();
        if let LearnedParams::Layer { params } = biases_params {
            return params;
        }
        panic!("Got invalid LearnedParams format for layer biases!");
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
        let biases: DataContainer = data.get_parameters();
        let momentum: DataContainer = data.get_momentum();
        let learning_rate: DataContainer = data.get_learning_rate();

        self.biases.borrow_mut().set_data(biases);
        if !matches!(&momentum, DataContainer::Empty) {
            self.biases.borrow_mut().set_momentum(momentum);
        }
        if !matches!(&learning_rate, DataContainer::Empty) {
            self.biases.borrow_mut().set_learning_rate(learning_rate);
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
        if !norm_params.is_null() {
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
        let data = self.biases.borrow_mut().get_data();

        if let DataContainer::Parameter(Data::VectorF32(vec)) = data {
            return vec.to_vec();
        }

        println!("Couldn't package biases for serialization due to invalid data container/type");
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
