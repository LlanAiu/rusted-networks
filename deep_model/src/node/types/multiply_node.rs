// builtin

// external
use ndarray::Array1;

// internal
use crate::node::{types::weight_node::WeightNode, Data, Node};

pub struct MultiplyNode<'a> {
    inputs: Vec<&'a dyn Node<'a>>,
    outputs: Vec<&'a dyn Node<'a>>,
    data: Data,
}

impl<'a> MultiplyNode<'a> {
    pub fn new(output_dim: usize) -> MultiplyNode<'a> {
        return MultiplyNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::VectorF32(Array1::zeros(output_dim)),
        };
    }

    fn add_weight_input(&mut self, weight: &'a WeightNode<'a>) {
        self.inputs.push(weight);
    }
}

impl<'a> Node<'a> for MultiplyNode<'a> {
    fn add_output(&mut self, output: &'a dyn Node<'a>) {
        self.outputs.push(output);
    }

    fn get_inputs(&self) -> &Vec<&dyn Node<'a>> {
        &self.inputs
    }

    fn get_outputs(&self) -> &Vec<&dyn Node<'a>> {
        &self.outputs
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn apply_operation(&self) {
        if self.inputs.len() != 2 {
            panic!(
                "Improper multiplicative node input number: expected 2 but got {}",
                self.inputs.len()
            );
        }
        let weights: &Data = self.inputs.get(0).unwrap().to_owned().get_data();

        if let Data::MatrixF32(data) = weights {
            todo!()
        } else {
            panic!(
                "Invalid weight type: expected Data::MatrixF32 but got {}",
                weights.variant_name()
            );
        }
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
