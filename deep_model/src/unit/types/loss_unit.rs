// builtin

// external

// internal
use crate::{
    data::Data,
    node::{
        types::{expected_response_node::ExpectedResponseNode, loss_node::LossNode},
        NodeRef,
    },
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct LossUnit<'a> {
    base: UnitBase<'a>,
    response_node: NodeRef<'a>,
}

impl<'a> LossUnit<'a> {
    pub fn new(output_dim: usize, loss_type: &str) -> LossUnit<'a> {
        let loss_ref: NodeRef = NodeRef::new(LossNode::new(loss_type));
        let response_ref: NodeRef = NodeRef::new(ExpectedResponseNode::new(output_dim));

        loss_ref.borrow_mut().add_input(&loss_ref, &response_ref);

        LossUnit {
            base: UnitBase::new(&loss_ref, &loss_ref),
            response_node: response_ref,
        }
    }

    pub fn set_expected_response(&self, response: Data) {
        self.response_node.borrow_mut().set_data(response);
    }
}

impl<'a> Unit<'a> for LossUnit<'a> {
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
