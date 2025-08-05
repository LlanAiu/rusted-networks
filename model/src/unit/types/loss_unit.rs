// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::config_types::loss_params::LossParams,
    node::{
        types::{
            add_node::AddNode, expected_response_node::ExpectedResponseNode, loss_node::LossNode,
        },
        NodeRef,
    },
    regularization::norm_penalty::l2_penalty::L2Ref,
    unit::{unit_base::UnitBase, Unit, UnitRef},
};

pub struct LossUnit<'a> {
    base: UnitBase<'a>,
    response_node: NodeRef<'a>,
    sum_node: NodeRef<'a>,
    loss_type: String,
    output_size: Vec<usize>,
}

impl<'a> LossUnit<'a> {
    pub fn new(output_dim: Vec<usize>, loss_type: &str) -> LossUnit<'a> {
        let loss_ref: NodeRef = NodeRef::new(LossNode::new(loss_type));
        let response_ref: NodeRef = NodeRef::new(ExpectedResponseNode::new(output_dim.clone()));
        let sum_ref: NodeRef = NodeRef::new(AddNode::with_print());

        loss_ref.borrow_mut().add_input(&loss_ref, &response_ref);
        sum_ref.borrow_mut().add_input(&sum_ref, &loss_ref);

        LossUnit {
            base: UnitBase::new(&loss_ref, &sum_ref),
            response_node: response_ref,
            sum_node: sum_ref,
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

    pub fn add_reg_node(&self, reg_unit: &L2Ref<'a>) {
        let sum_ref = &self.sum_node;

        sum_ref
            .borrow_mut()
            .add_input(sum_ref, reg_unit.borrow().get_output_ref());
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
