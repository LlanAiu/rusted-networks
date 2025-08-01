// builtin
use std::{cell::RefCell, rc::Rc};

// external

// internal
use crate::node::{
    types::{add_node::AddNode, element_sum_node::ElementSumNode, square_node::SquareNode},
    NodeRef,
};

pub type L2Ref<'a> = Rc<RefCell<L2PenaltyUnit<'a>>>;

pub struct L2PenaltyUnit<'a> {
    weight_input: NodeRef<'a>,
    penalty_input: NodeRef<'a>,
    penalty_output: NodeRef<'a>,
}

impl<'a> L2PenaltyUnit<'a> {
    pub fn new() -> L2PenaltyUnit<'a> {
        let square = NodeRef::new(SquareNode::new());
        let element_sum = NodeRef::new(ElementSumNode::new());
        let add = NodeRef::new(AddNode::new());

        add.borrow_mut().add_input(&add, &element_sum);
        element_sum.borrow_mut().add_input(&element_sum, &square);

        L2PenaltyUnit {
            weight_input: square,
            penalty_input: add.clone(),
            penalty_output: add,
        }
    }

    pub fn add_penalty_input(&mut self, input: &L2Ref<'a>) {
        let penalty_input = &self.penalty_input;
        penalty_input
            .borrow_mut()
            .add_input(&penalty_input, input.borrow().get_output_ref());
    }

    pub fn add_weight_input(&mut self, weight_node: &NodeRef<'a>) {
        let weight_input = &self.weight_input;
        weight_input
            .borrow_mut()
            .add_input(&weight_input, weight_node);
    }

    pub fn get_output_ref(&self) -> &NodeRef<'a> {
        &self.penalty_output
    }
}
