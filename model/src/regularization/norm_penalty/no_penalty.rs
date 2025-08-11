// builtin

// external

// internal
use crate::{
    node::NodeRef,
    regularization::norm_penalty::{NormPenaltyRef, NormPenaltyUnit},
};
pub mod builder;

pub struct NullPenaltyUnit;

impl NullPenaltyUnit {
    pub fn new() -> NullPenaltyUnit {
        NullPenaltyUnit
    }
}

impl<'a> NormPenaltyUnit<'a> for NullPenaltyUnit {
    fn add_penalty_input(&mut self, _input: &NormPenaltyRef<'a>) {
        println!("[NULL] Attempted to add penalty input to null penalty node");
    }

    fn add_parameter_input(&mut self, _parameter_node: &NodeRef<'a>) {
        println!("[NULL] Attempted to add parameter input to null penalty node");
    }

    fn get_output_ref(&self) -> &NodeRef<'a> {
        panic!("[NULL] Null penalty node has no output node, try enforcing a null check penalty_unit.is_null()");
    }

    fn is_null(&self) -> bool {
        true
    }
}
