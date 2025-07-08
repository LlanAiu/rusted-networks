// builtin
use std::{cell::RefCell, rc::Rc};

// external

// internal
use crate::data::Data;
pub mod activation;
pub mod node_base;
pub mod types;

pub type NodeRef<'a> = Rc<RefCell<dyn Node<'a> + 'a>>;

pub trait Node<'a> {
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
