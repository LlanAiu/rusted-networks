// builtin

// external

// internal

use std::rc::Rc;

use crate::{node::NodeRef, unit::UnitRef};

pub struct UnitBase<'a> {
    inputs: Vec<UnitRef<'a>>,
    outputs: Vec<UnitRef<'a>>,
    input_node: NodeRef<'a>,
    output_node: NodeRef<'a>,
}

impl<'a> UnitBase<'a> {
    pub fn new(input: &NodeRef<'a>, output: &NodeRef<'a>) -> UnitBase<'a> {
        UnitBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_node: NodeRef::clone(input),
            output_node: NodeRef::clone(output),
        }
    }

    pub fn add_input(&mut self, this: &UnitRef<'a>, input: &UnitRef<'a>) {
        input.borrow_mut().add_output(this);
        self.inputs.push(Rc::clone(input));

        self.input_node
            .borrow_mut()
            .add_input(&self.input_node, input.borrow().get_output_node());
    }

    pub fn add_output(&mut self, output: &UnitRef<'a>) {
        self.outputs.push(Rc::clone(output));
    }

    pub fn get_inputs(&self) -> &Vec<UnitRef<'a>> {
        &self.inputs
    }

    pub fn get_outputs(&self) -> &Vec<UnitRef<'a>> {
        &self.outputs
    }

    pub fn get_output_node(&self) -> &NodeRef<'a> {
        &self.output_node
    }
}
