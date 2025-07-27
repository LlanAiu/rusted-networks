// builtin

// external

use core::panic;
use std::usize;

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    network::config_types::UnitParams,
    node::{
        types::{
            activation_node::ActivationNode, add_node::AddNode, bias_node::BiasNode,
            matrix_multiply_node::MatrixMultiplyNode, weight_node::WeightNode,
        },
        NodeRef,
    },
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct LinearUnit<'a> {
    base: UnitBase<'a>,
    weights: NodeRef<'a>,
    biases: NodeRef<'a>,
    input_size: usize,
    output_size: usize,
    activation: String,
}

impl<'a> LinearUnit<'a> {
    pub fn new(
        function: &str,
        input_size: usize,
        output_size: usize,
        learning_rate: f32,
    ) -> LinearUnit<'a> {
        let weights_ref: NodeRef =
            NodeRef::new(WeightNode::new(input_size, output_size, learning_rate));
        let biases_ref: NodeRef = NodeRef::new(BiasNode::new(output_size, learning_rate));
        let matmul_ref: NodeRef = NodeRef::new(MatrixMultiplyNode::new());
        let add_ref: NodeRef = NodeRef::new(AddNode::new());
        let activation_ref: NodeRef = NodeRef::new(ActivationNode::new(function));

        matmul_ref.borrow_mut().add_input(&matmul_ref, &weights_ref);

        add_ref.borrow_mut().add_input(&add_ref, &matmul_ref);
        add_ref.borrow_mut().add_input(&add_ref, &biases_ref);

        activation_ref
            .borrow_mut()
            .add_input(&activation_ref, &add_ref);

        LinearUnit {
            base: UnitBase::new(&matmul_ref, &activation_ref),
            weights: weights_ref,
            biases: biases_ref,
            input_size,
            output_size,
            activation: function.to_string(),
        }
    }

    pub fn from_config(config: &UnitParams, learning_rate: f32) -> LinearUnit<'a> {
        if let UnitParams::Linear {
            input_size,
            output_size,
            activation,
            ..
        } = config
        {
            let unit: LinearUnit = Self::new(activation, *input_size, *output_size, learning_rate);

            let weights = config.get_weights();
            let biases = config.get_biases();

            unit.set_weights(weights);
            unit.set_biases(biases);

            return unit;
        }

        panic!("Mismatched unit parameter types for initialization: expected UnitParams::Linear but got {},", config.type_name());
    }

    pub fn set_biases(&self, data: DataContainer) {
        self.biases.borrow_mut().set_data(data);
    }

    pub fn set_weights(&self, data: DataContainer) {
        self.weights.borrow_mut().set_data(data);
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
}
