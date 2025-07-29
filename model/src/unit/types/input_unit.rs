// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::input_params::InputParams,
    node::{types::input_node::InputNode, NodeRef},
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct InputUnit<'a> {
    base: UnitBase<'a>,
    input: NodeRef<'a>,
    input_size: Vec<usize>,
}

impl<'a> InputUnit<'a> {
    pub fn new(input_size: Vec<usize>) -> InputUnit<'a> {
        let input_ref = NodeRef::new(InputNode::new(input_size.clone()));

        InputUnit {
            base: UnitBase::new(&input_ref, &input_ref),
            input: input_ref,
            input_size,
        }
    }

    pub fn from_config(config: &InputParams) -> InputUnit<'a> {
        Self::new(config.input_size.clone())
    }

    pub fn set_input_data(&self, data: DataContainer) {
        self.input.borrow_mut().set_data(data);
    }

    pub fn get_input_size(&self) -> &[usize] {
        &self.input_size
    }
}

impl<'a> Unit<'a> for InputUnit<'a> {
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
