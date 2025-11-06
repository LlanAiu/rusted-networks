// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::input_params::InputParams,
    node::{
        types::{input_node::InputNode, mask_node::MaskNode, multiply_node::MultiplyNode},
        NodeRef,
    },
    regularization::dropout::{NetworkMode, UnitMaskType},
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct InputUnit<'a> {
    base: UnitBase<'a>,
    input: NodeRef<'a>,
    input_size: Vec<usize>,
    mask_type: UnitMaskType,
}

impl<'a> InputUnit<'a> {
    pub fn new(input_size: Vec<usize>, mask_type: UnitMaskType) -> InputUnit<'a> {
        let input_ref = NodeRef::new(InputNode::new(input_size.clone()));

        let mut output_ref: &NodeRef = &input_ref;

        let mut mask: Option<&NodeRef> = Option::None;

        let mask_ref: NodeRef;
        let multiply_ref: NodeRef;

        if let UnitMaskType::Dropout { keep_probability } = &mask_type {
            mask_ref = NodeRef::new(MaskNode::new(input_size.clone(), *keep_probability));
            multiply_ref = NodeRef::new(MultiplyNode::new());

            multiply_ref
                .borrow_mut()
                .add_input(&multiply_ref, &input_ref);

            multiply_ref
                .borrow_mut()
                .add_input(&multiply_ref, &mask_ref);

            mask = Option::Some(&mask_ref);
            output_ref = &multiply_ref
        }

        InputUnit {
            base: UnitBase::new(&input_ref, &output_ref, mask, false),
            input: input_ref,
            input_size,
            mask_type,
        }
    }

    pub fn from_config(config: &InputParams) -> InputUnit<'a> {
        Self::new(
            config.get_input_size().clone(),
            UnitMaskType::from_keep_probability(config.get_keep_probability()),
        )
    }

    pub fn set_input_data(&self, data: DataContainer) {
        self.input.borrow_mut().set_data(data);
    }

    pub fn get_input_size(&self) -> &[usize] {
        &self.input_size
    }

    pub fn get_mask_type(&self) -> &UnitMaskType {
        &self.mask_type
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

    fn update_mode(&mut self, new_mode: NetworkMode) {
        self.base.update_mode(new_mode);

        for unit in self.base.get_outputs() {
            unit.borrow_mut().update_mode(new_mode);
        }
    }
}
