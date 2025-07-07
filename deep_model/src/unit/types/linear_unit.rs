// builtin

// external

// internal

use std::{cell::RefCell, rc::Rc};

use crate::{
    node::{
        types::{
            activation_node::ActivationNode, add_node::AddNode, bias_node::BiasNode,
            multiply_node::MultiplyNode, weight_node::WeightNode,
        },
        Data, NodeRef,
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
        let weights: NodeRef = Rc::new(RefCell::new(WeightNode::new(
            input_size,
            output_size,
            0.001,
        )));
        let biases: NodeRef = Rc::new(RefCell::new(BiasNode::new(output_size, 0.001)));

        let multiply: NodeRef = Rc::new(RefCell::new(MultiplyNode::new()));
        let add: NodeRef = Rc::new(RefCell::new(AddNode::new()));
        let activation: NodeRef = Rc::new(RefCell::new(ActivationNode::new(function)));

        multiply.borrow_mut().add_input(&multiply, &weights);

        add.borrow_mut().add_input(&add, &multiply);
        add.borrow_mut().add_input(&add, &biases);

        activation.borrow_mut().add_input(&activation, &add);

        LinearUnit {
            base: UnitBase::new(&multiply, &activation),
            weights,
            biases,
        }
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

    fn set_biases(&mut self, data: Data) {
        self.biases.borrow_mut().set_data(data);
    }

    fn set_weights(&mut self, data: Data) {
        self.weights.borrow_mut().set_data(data);
    }

    fn set_input_data(&mut self, _data: Data) {}
}
