// builtin

// external

// internal
use crate::data::data_container::DataContainer;
use crate::data::Data;
use crate::network::config_types::learned_params::LearnedParams;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};
use crate::regularization::dropout::NetworkMode;

pub struct ConstantNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> ConstantNode<'a> {
    pub fn new(data: Data) -> ConstantNode<'a> {
        let mut base = NodeBase::new();

        base.set_data(DataContainer::Parameter(data));

        return ConstantNode { base };
    }
}

impl<'a> Node<'a> for ConstantNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Parameter
    }

    fn add_input(&mut self, _this: &NodeRef<'a>, _input: &NodeRef<'a>) {}

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.base.add_output(output);
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_outputs()
    }

    fn set_data(&mut self, _input: DataContainer) {}

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {}

    fn add_gradient(&mut self, _grad: &DataContainer) {
        self.base.increment_grad_count();
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }

    fn set_momentum(&mut self, _momentum: DataContainer) {
        println!("[CONSTANT] Unsupported Operation: Cannot set momentum of a constant node");
    }

    fn set_learning_rate(&mut self, _learning_rate: DataContainer) {
        println!("[CONSTANT] Unsupported Operation: Cannot set learning rate of a constant node");
    }

    fn save_parameters(&self) -> LearnedParams {
        println!("[CONSTANT] Unsupported Operation: Cannot save parameters of a constant node");
        LearnedParams::null()
    }

    fn set_mode(&mut self, _new_mode: NetworkMode) {}
}
