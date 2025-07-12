// builtin
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Display,
    rc::Rc,
};

// external

// internal
use crate::data::Data;
pub mod activation;
pub mod loss;
pub mod node_base;
pub mod types;

// pub type NodeRef<'a> = Rc<RefCell<dyn Node<'a> + 'a>>;

#[derive(Clone)]
pub struct NodeRef<'a> {
    reference: Rc<RefCell<dyn Node<'a> + 'a>>,
    node_type: NodeType,
}

impl<'a> NodeRef<'a> {
    pub fn new(node: Rc<RefCell<dyn Node<'a> + 'a>>) -> NodeRef<'a> {
        let node_type: NodeType = node.borrow().get_type();
        NodeRef {
            reference: node,
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

    fn set_data(&mut self, data: Data);

    fn get_data(&mut self) -> Data;

    fn apply_operation(&mut self);

    fn add_gradient(&mut self, grad: &Data);

    fn apply_jacobian(&mut self);

    fn should_process_backprop(&self) -> bool;
}
