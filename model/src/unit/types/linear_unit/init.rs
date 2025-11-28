// builtin

// external

// internal
use crate::{
    network::config_types::{batch_norm_params::NormParams, unit_params::UnitParams},
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
    regularization::dropout::UnitMaskType,
    unit::{types::linear_unit::LinearUnit, unit_base::UnitBase},
};

pub fn build_linear_unit_from_config<'a>(
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
        let unit: LinearUnit = create_linear_unit(
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

fn create_linear_unit<'a>(
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
    let batch_norm_enabled: bool = normalization_type.is_batch_norm_enabled() && !is_last_layer;
    let dropout_enabled: bool = mask_type.is_dropout_enabled() && !is_last_layer;
    let mut output_ref: NodeRef;

    let weights_ref: NodeRef = NodeRef::new(WeightNode::new_matrix(
        input_size,
        output_size,
        decay_type.clone(),
        descent_type.clone(),
    ));
    let matmul_ref: NodeRef = NodeRef::new(MatrixMultiplyNode::new());
    matmul_ref.borrow_mut().add_input(&matmul_ref, &weights_ref);
    output_ref = NodeRef::clone(&matmul_ref);

    let mut biases: Option<NodeRef> = Option::None;
    if !batch_norm_enabled {
        let (biases_ref, out_ref) = create_biases(
            NodeRef::clone(&output_ref),
            output_size,
            &decay_type,
            &descent_type,
        );
        biases = Option::Some(biases_ref);
        output_ref = NodeRef::clone(&out_ref);
    }

    let mut norm_module: Option<BatchNormModule> = Option::None;
    let mut norm: Option<NodeRef> = Option::None;
    if batch_norm_enabled {
        let (module, norm_ref, out_ref) = create_batch_norm(
            NodeRef::clone(&output_ref),
            &normalization_type,
            norm_params,
            output_size,
            &decay_type,
            &descent_type,
        );
        output_ref = NodeRef::clone(&out_ref);
        norm_module = Option::Some(module);
        norm = Option::Some(norm_ref);
    }

    let activation_ref: NodeRef = NodeRef::new(ActivationNode::new(function));
    activation_ref
        .borrow_mut()
        .add_input(&activation_ref, &output_ref);
    output_ref = NodeRef::clone(&activation_ref);

    let mut mask: Option<NodeRef> = Option::None;
    if dropout_enabled {
        let (mask_ref, out_ref) =
            create_dropout(NodeRef::clone(&output_ref), &mask_type, output_size);
        mask = Option::Some(mask_ref);
        output_ref = NodeRef::clone(&out_ref);
    }

    LinearUnit {
        base: UnitBase::new(matmul_ref, output_ref, mask, norm, is_last_layer),
        weights: weights_ref,
        biases,
        input_size,
        output_size,
        activation: function.to_string(),
        norm_module,
        mask_type,
    }
}

fn create_biases<'a>(
    prev_output: NodeRef<'a>,
    size: usize,
    decay_type: &LearningDecayType,
    descent_type: &DescentType,
) -> (NodeRef<'a>, NodeRef<'a>) {
    let biases_ref: NodeRef = NodeRef::new(BiasNode::new(
        size,
        decay_type.clone(),
        descent_type.clone(),
    ));
    let add_ref: NodeRef = NodeRef::new(AddNode::new());

    add_ref.borrow_mut().add_input(&add_ref, &prev_output);
    add_ref.borrow_mut().add_input(&add_ref, &biases_ref);

    (biases_ref, add_ref)
}

fn create_batch_norm<'a>(
    output: NodeRef<'a>,
    normalization_type: &NormalizationType,
    norm_params: &NormParams,
    size: usize,
    decay_type: &LearningDecayType,
    descent_type: &DescentType,
) -> (BatchNormModule<'a>, NodeRef<'a>, NodeRef<'a>) {
    if let NormalizationType::BatchNorm { decay } = normalization_type {
        let norm_ref: NodeRef;
        if norm_params.is_null() {
            norm_ref = NodeRef::new(NormalizationNode::new(*decay));
        } else {
            let mean = norm_params.get_mean();
            let variance = norm_params.get_variance();
            let decay = norm_params.get_decay();
            norm_ref = NodeRef::new(NormalizationNode::from_parameters(mean, variance, decay));
        }

        let scale_ref = NodeRef::new(WeightNode::new_vec(
            size,
            decay_type.clone(),
            descent_type.clone(),
        ));
        let shift_ref = NodeRef::new(BiasNode::new(
            size,
            decay_type.clone(),
            descent_type.clone(),
        ));
        let norm_multiply_ref: NodeRef = NodeRef::new(MultiplyNode::new());
        let norm_add_ref: NodeRef = NodeRef::new(AddNode::new());

        norm_ref.borrow_mut().add_input(&norm_ref, &output);

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

        let module = BatchNormModule::new(&norm_ref, &scale_ref, &shift_ref);
        return (module, norm_ref, norm_add_ref);
    }

    panic!("Tried to create batch norm module with a NormalizationType::None configuration!");
}

fn create_dropout<'a>(
    output: NodeRef<'a>,
    mask_type: &UnitMaskType,
    size: usize,
) -> (NodeRef<'a>, NodeRef<'a>) {
    if let UnitMaskType::Dropout {
        keep_probability: probability,
    } = mask_type
    {
        let mask_ref: NodeRef = NodeRef::new(MaskNode::new(vec![size], *probability));
        let multiply_ref: NodeRef = NodeRef::new(MultiplyNode::new());

        multiply_ref.borrow_mut().add_input(&multiply_ref, &output);

        multiply_ref
            .borrow_mut()
            .add_input(&multiply_ref, &mask_ref);

        return (mask_ref, multiply_ref);
    }

    panic!("Tried to create dropout module with a UnitMaskType::None configuration!");
}
