// builtin
use std::{cell::RefCell, rc::Rc};

// external

// internal
use crate::node::{Data, NodeRef};
pub mod types;
pub mod unit_base;

pub type UnitRef<'a> = Rc<RefCell<dyn Unit<'a> + 'a>>;

pub trait Unit<'a> {
    fn add_input(&mut self, this: &UnitRef<'a>, input: &UnitRef<'a>);

    fn add_output(&mut self, output: &UnitRef<'a>);

    fn get_inputs(&self) -> &Vec<UnitRef<'a>>;

    fn get_outputs(&self) -> &Vec<UnitRef<'a>>;

    fn get_output_node(&self) -> &NodeRef<'a>;

    fn set_data(&mut self, data: Data);

    fn get_data(&mut self) -> Data;

    fn feedforward(&mut self);

    fn get_jacobian(&self);
}
