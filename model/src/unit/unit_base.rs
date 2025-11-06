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
    mode: NetworkMode,
    is_inference: bool,
}

impl<'a> UnitBase<'a> {
    pub fn new(
        input: &NodeRef<'a>,
        output: &NodeRef<'a>,
        mask: Option<&NodeRef<'a>>,
        is_inference: bool,
    ) -> UnitBase<'a> {
        let mut mask_node: Option<NodeRef<'a>> = Option::None;

        if let Option::Some(node) = mask {
            mask_node = Option::Some(NodeRef::clone(node))
        }

        UnitBase {
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_node: NodeRef::clone(input),
            output_node: NodeRef::clone(output),
            mask_node,
            is_inference,
            mode: if is_inference {
                NetworkMode::Inference
            } else {
                NetworkMode::None
            },
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

    pub fn is_inference(&self) -> bool {
        self.is_inference
    }

    pub fn update_mode(&mut self, new_mode: NetworkMode) {
        if self.is_inference {
            return;
        }

        if new_mode != self.mode {
            self.mode = new_mode;

            if let Option::Some(mask) = &self.mask_node {
                mask.borrow_mut().set_mode(new_mode)
            }
        }
    }
}
