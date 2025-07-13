// builtin

// external

// internal

use std::{cell::RefCell, rc::Rc};

use crate::{
    data::Data,
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
}

impl<'a> LinearUnit<'a> {
    pub fn new(function: &str, input_size: usize, output_size: usize) -> LinearUnit<'a> {
        let weights = Rc::new(RefCell::new(WeightNode::new(
            input_size,
            output_size,
            0.001,
        )));
        let biases = Rc::new(RefCell::new(BiasNode::new(output_size, 0.001)));
        let matmul = Rc::new(RefCell::new(MatrixMultiplyNode::new()));
        let add = Rc::new(RefCell::new(AddNode::new()));
        let activation = Rc::new(RefCell::new(ActivationNode::new(function)));

        let weights_ref: NodeRef = NodeRef::new(weights);
        let biases_ref: NodeRef = NodeRef::new(biases);
        let matmul_ref: NodeRef = NodeRef::new(matmul);
        let add_ref: NodeRef = NodeRef::new(add);
        let activation_ref: NodeRef = NodeRef::new(activation);

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
        }
    }

    pub fn set_biases(&self, data: Data) {
        self.biases.borrow_mut().set_data(data);
    }

    pub fn set_weights(&self, data: Data) {
        self.weights.borrow_mut().set_data(data);
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
