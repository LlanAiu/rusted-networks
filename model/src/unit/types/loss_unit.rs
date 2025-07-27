// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::LossParams,
    node::{
        types::{expected_response_node::ExpectedResponseNode, loss_node::LossNode},
        NodeRef,
    },
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct LossUnit<'a> {
    base: UnitBase<'a>,
    response_node: NodeRef<'a>,
    loss_type: String,
    output_size: Vec<usize>,
}

impl<'a> LossUnit<'a> {
    pub fn new(output_dim: Vec<usize>, loss_type: &str) -> LossUnit<'a> {
        let loss_ref: NodeRef = NodeRef::new(LossNode::new(loss_type));
        let response_ref: NodeRef = NodeRef::new(ExpectedResponseNode::new(output_dim.clone()));

        loss_ref.borrow_mut().add_input(&loss_ref, &response_ref);

        LossUnit {
            base: UnitBase::new(&loss_ref, &loss_ref),
            response_node: response_ref,
            loss_type: loss_type.to_string(),
            output_size: output_dim,
        }
    }

    pub fn from_config(config: &LossParams) -> LossUnit<'a> {
        Self::new(config.output_size.clone(), &config.loss_type)
    }

    pub fn set_expected_response(&self, response: DataContainer) {
        self.response_node.borrow_mut().set_data(response);
    }

    pub fn get_loss_type(&self) -> &str {
        &self.loss_type
    }

    pub fn get_output_size(&self) -> &[usize] {
        &self.output_size
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
