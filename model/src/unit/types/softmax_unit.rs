// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::config_types::{learned_params::LearnedParams, unit_params::UnitParams},
    node::{
        types::{
            activation_node::ActivationNode, add_node::AddNode, bias_node::BiasNode,
            matrix_multiply_node::MatrixMultiplyNode, softmax_node::SoftmaxNode,
            weight_node::WeightNode,
        },
        NodeRef,
    },
    optimization::{learning_decay::LearningDecayType, momentum::DescentType},
    regularization::dropout::NetworkMode,
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct SoftmaxUnit<'a> {
    base: UnitBase<'a>,
    weights: NodeRef<'a>,
    biases: NodeRef<'a>,
    input_size: usize,
    output_size: usize,
    activation: String,
}

impl<'a> SoftmaxUnit<'a> {
    // TODO: update config to handle dropout parameters
    pub fn new(
        function: &str,
        input_size: usize,
        output_size: usize,
        decay_type: LearningDecayType,
        descent_type: DescentType,
        is_inference: bool,
    ) -> SoftmaxUnit<'a> {
        let weights_ref: NodeRef = NodeRef::new(WeightNode::new(
            input_size,
            output_size,
            decay_type.clone(),
            descent_type.clone(),
        ));
        let biases_ref = NodeRef::new(BiasNode::new(output_size, decay_type, descent_type));
        let matmul_ref: NodeRef = NodeRef::new(MatrixMultiplyNode::new());
        let add_ref: NodeRef = NodeRef::new(AddNode::new());
        let activation_ref: NodeRef = NodeRef::new(ActivationNode::new(function));
        let softmax_ref: NodeRef = NodeRef::new(SoftmaxNode::new());

        matmul_ref.borrow_mut().add_input(&matmul_ref, &weights_ref);

        add_ref.borrow_mut().add_input(&add_ref, &matmul_ref);
        add_ref.borrow_mut().add_input(&add_ref, &biases_ref);

        activation_ref
            .borrow_mut()
            .add_input(&activation_ref, &add_ref);

        softmax_ref
            .borrow_mut()
            .add_input(&softmax_ref, &activation_ref);

        SoftmaxUnit {
            // TODO: Swap out the Option::None for a potential dropout mask node
            base: UnitBase::new(&matmul_ref, &softmax_ref, Option::None, is_inference),
            weights: weights_ref,
            biases: biases_ref,
            input_size,
            output_size,
            activation: function.to_string(),
        }
    }

    pub fn from_config(
        config: &UnitParams,
        decay_type: LearningDecayType,
        descent_type: DescentType,
    ) -> SoftmaxUnit<'a> {
        if let UnitParams::Softmax {
            input_size,
            output_size,
            activation,
            weights,
            biases,
            is_inference,
        } = config
        {
            let unit: SoftmaxUnit = Self::new(
                activation,
                *input_size,
                *output_size,
                decay_type,
                descent_type,
                *is_inference,
            );

            unit.set_weights(weights);
            unit.set_biases(biases);

            return unit;
        }

        panic!("Mismatched unit parameter types for initialization: expected UnitParams::Softmax but got {},", config.type_name());
    }

    pub fn get_weights_params(&self) -> LearnedParams {
        self.weights.borrow().save_parameters()
    }

    pub fn get_biases_params(&self) -> LearnedParams {
        self.biases.borrow().save_parameters()
    }

    pub fn get_weights_ref(&self) -> &NodeRef<'a> {
        &self.weights
    }

    pub fn set_biases(&self, data: &LearnedParams) {
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

    pub fn set_weights(&self, data: &LearnedParams) {
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

    pub fn get_input_size(&self) -> usize {
        self.input_size
    }

    pub fn get_output_size(&self) -> usize {
        self.output_size
    }

    pub fn get_weights(&self) -> Vec<f32> {
        let data = self.weights.borrow_mut().get_data();

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

    pub fn is_inference(&self) -> bool {
        self.base.is_inference()
    }
}

impl<'a> Unit<'a> for SoftmaxUnit<'a> {
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
