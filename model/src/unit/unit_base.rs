// builtin

// external

// internal

use std::rc::Rc;

use crate::{node::NodeRef, regularization::dropout::NetworkMode, unit::UnitRef};

pub struct UnitBase<'a> {
    inputs: Vec<UnitRef<'a>>,
    outputs: Vec<UnitRef<'a>>,
    input_node: NodeRef<'a>,
    output_node: NodeRef<'a>,
    mask_node: Option<NodeRef<'a>>,
    norm_node: Option<NodeRef<'a>>,
    mode: NetworkMode,
    is_last_layer: bool,
}

impl<'a> UnitBase<'a> {
    pub fn new(
        input: NodeRef<'a>,
        output: NodeRef<'a>,
        mask: Option<NodeRef<'a>>,
        norm: Option<NodeRef<'a>>,
        is_last_layer: bool,
    ) -> UnitBase<'a> {
        let mut mask_node: Option<NodeRef<'a>> = Option::None;
        if let Option::Some(mask_ref) = mask {
            mask_node = Option::Some(mask_ref)
        }

        let mut norm_node: Option<NodeRef<'a>> = Option::None;
        if let Option::Some(norm_ref) = norm {
            norm_node = Option::Some(norm_ref)
        }

        UnitBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_node: input,
            output_node: output,
            mask_node,
            norm_node,
            is_last_layer,
            mode: NetworkMode::None,
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

    pub fn is_last_layer(&self) -> bool {
        self.is_last_layer
    }

    pub fn update_mode(&mut self, new_mode: NetworkMode) {
        if new_mode != self.mode {
            self.mode = new_mode;

            if let Option::Some(mask) = &self.mask_node {
                mask.borrow_mut().set_mode(new_mode)
            }

            if let Option::Some(norm) = &self.norm_node {
                norm.borrow_mut().set_mode(new_mode)
            }
        }
    }
}
