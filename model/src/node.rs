// builtin
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
    rc::Rc,
};

// external

// internal
use crate::{
    data::data_container::DataContainer, network::config_types::learned_params::LearnedParams,
    regularization::dropout::NetworkMode,
};
pub mod activation;
pub mod loss;
pub mod node_base;
pub mod types;

#[derive(Clone)]
pub struct NodeRef<'a> {
    reference: Rc<RefCell<dyn Node<'a> + 'a>>,
    node_type: NodeType,
}

impl<'a> NodeRef<'a> {
    pub fn new(node: impl Node<'a> + 'a) -> NodeRef<'a> {
        let node_type: NodeType = node.get_type();
        NodeRef {
            reference: Rc::new(RefCell::new(node)),
            node_type,
        }
    }

    pub fn clone(node: &NodeRef<'a>) -> NodeRef<'a> {
        NodeRef {
            reference: node.get_reference(),
            node_type: node.get_type(),
        }
    }

    pub fn get_reference(&self) -> Rc<RefCell<dyn Node<'a> + 'a>> {
        Rc::clone(&self.reference)
    }

    pub fn get_type(&self) -> NodeType {
        self.node_type
    }

    pub fn borrow(&self) -> Ref<'_, dyn Node<'a> + 'a> {
        self.reference.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, dyn Node<'a> + 'a> {
        self.reference.borrow_mut()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Parameter,
    Input,
    ExpectedResponse,
    Operation,
    None,
}

impl Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            NodeType::Parameter => "Parameter",
            NodeType::Input => "Input",
            NodeType::ExpectedResponse => "ExpectedResponse",
            NodeType::Operation => "Operation",
            NodeType::None => "None",
        };
        write!(f, "{}", s)
    }
}

pub trait Node<'a> {
    fn get_type(&self) -> NodeType;

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>);

    fn add_output(&mut self, output: &NodeRef<'a>);

    fn get_inputs(&self) -> &Vec<NodeRef<'a>>;

    fn get_outputs(&self) -> &Vec<NodeRef<'a>>;

    fn get_data(&mut self) -> DataContainer;

    fn save_parameters(&self) -> LearnedParams;

    fn set_data(&mut self, data: DataContainer);

    fn set_momentum(&mut self, momentum: DataContainer);

    fn set_mode(&mut self, new_mode: NetworkMode);

    fn set_learning_rate(&mut self, learning_rate: DataContainer);

    fn apply_operation(&mut self);

    fn add_gradient(&mut self, grad: &DataContainer);

    fn apply_jacobian(&mut self);

    fn should_process_backprop(&self) -> bool;
}
