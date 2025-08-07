// builtin
use std::{cell::RefCell, rc::Rc};

// external

// internal
use crate::{
    data::Data,
    node::{
        types::{
            add_node::AddNode, constant_node::ConstantNode, element_sum_node::ElementSumNode,
            multiply_node::MultiplyNode, square_node::SquareNode,
        },
        NodeRef,
    },
};

pub type L2Ref<'a> = Rc<RefCell<L2PenaltyUnit<'a>>>;

pub struct L2PenaltyUnit<'a> {
    weight_input: NodeRef<'a>,
    penalty_input: NodeRef<'a>,
    penalty_output: NodeRef<'a>,
}

impl<'a> L2PenaltyUnit<'a> {
    pub fn new(scale: f32) -> L2PenaltyUnit<'a> {
        let square = NodeRef::new(SquareNode::new());
        let element_sum = NodeRef::new(ElementSumNode::new());
        let add = NodeRef::new(AddNode::new());
        let multiply = NodeRef::new(MultiplyNode::new());
        let scale = NodeRef::new(ConstantNode::new(Data::ScalarF32(scale)));

        add.borrow_mut().add_input(&add, &multiply);
        multiply.borrow_mut().add_input(&multiply, &element_sum);
        multiply.borrow_mut().add_input(&multiply, &scale);
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

#[cfg(test)]
mod tests {
    use crate::{
        data::data_container::DataContainer,
        node::{types::weight_node::WeightNode, NodeRef},
        regularization::norm_penalty::l2_penalty::L2PenaltyUnit,
    };

    #[test]
    fn penalty_test() {
        let mut unit: L2PenaltyUnit = L2PenaltyUnit::new(1.0);
        let weight: NodeRef = NodeRef::new(WeightNode::new(4, 2, 0.001));

        println!("Weights: {:?}", weight.borrow_mut().get_data());

        unit.add_weight_input(&weight);

        unit.get_output_ref().borrow_mut().apply_operation();

        let output = unit.get_output_ref().borrow_mut().get_data();

        println!("Output: {:?}", output);
    }

    #[test]
    fn penalty_backprop() {
        let mut unit: L2PenaltyUnit = L2PenaltyUnit::new(1.0);
        let weight: NodeRef = NodeRef::new(WeightNode::new(4, 2, 1.0));

        let old_weights = weight.borrow_mut().get_data();
        println!("Weights: {:?}", old_weights);

        unit.add_weight_input(&weight);

        let output_ref = unit.get_output_ref();
        output_ref.borrow_mut().apply_operation();
        output_ref.borrow_mut().add_gradient(&DataContainer::one());
        output_ref.borrow_mut().apply_jacobian();

        let new_weights = weight.borrow_mut().get_data();
        println!("New weights: {:?}", new_weights);
    }
}
