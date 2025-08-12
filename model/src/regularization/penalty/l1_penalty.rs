// builtin

// external

// internal
use crate::{
    data::Data,
    node::{
        types::{
            abs_value_node::AbsoluteValueNode, add_node::AddNode, constant_node::ConstantNode,
            element_sum_node::ElementSumNode, multiply_node::MultiplyNode,
        },
        NodeRef,
    },
    regularization::penalty::{PenaltyRef, PenaltyUnit},
};
pub mod builder;

pub struct L1PenaltyUnit<'a> {
    weight_input: NodeRef<'a>,
    penalty_input: NodeRef<'a>,
    penalty_output: NodeRef<'a>,
}

impl<'a> L1PenaltyUnit<'a> {
    pub fn new(scale: f32) -> L1PenaltyUnit<'a> {
        let absolute = NodeRef::new(AbsoluteValueNode::new());
        let element_sum = NodeRef::new(ElementSumNode::new());
        let add = NodeRef::new(AddNode::new());
        let multiply = NodeRef::new(MultiplyNode::new());
        let scale = NodeRef::new(ConstantNode::new(Data::ScalarF32(scale)));

        add.borrow_mut().add_input(&add, &multiply);
        multiply.borrow_mut().add_input(&multiply, &element_sum);
        multiply.borrow_mut().add_input(&multiply, &scale);
        element_sum.borrow_mut().add_input(&element_sum, &absolute);

        L1PenaltyUnit {
            weight_input: absolute,
            penalty_input: add.clone(),
            penalty_output: add,
        }
    }
}

impl<'a> PenaltyUnit<'a> for L1PenaltyUnit<'a> {
    fn add_penalty_input(&mut self, input: &PenaltyRef<'a>) {
        let penalty_input = &self.penalty_input;
        penalty_input
            .borrow_mut()
            .add_input(&penalty_input, input.borrow().get_output_ref());
    }

    fn add_parameter_input(&mut self, weight_node: &NodeRef<'a>) {
        let weight_input = &self.weight_input;
        weight_input
            .borrow_mut()
            .add_input(&weight_input, weight_node);
    }

    fn get_output_ref(&self) -> &NodeRef<'a> {
        &self.penalty_output
    }

    fn is_null(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        data::data_container::DataContainer,
        node::{types::weight_node::WeightNode, NodeRef},
        regularization::penalty::l1_penalty::L1PenaltyUnit,
        regularization::penalty::PenaltyUnit,
    };

    #[test]
    fn penalty_test() {
        let mut unit: L1PenaltyUnit = L1PenaltyUnit::new(1.0);
        let weight: NodeRef = NodeRef::new(WeightNode::new(4, 2, 0.001));

        println!("Weights: {:?}", weight.borrow_mut().get_data());

        unit.add_parameter_input(&weight);

        unit.get_output_ref().borrow_mut().apply_operation();

        let output = unit.get_output_ref().borrow_mut().get_data();

        println!("Output: {:?}", output);
    }

    #[test]
    fn penalty_backprop() {
        let mut unit: L1PenaltyUnit = L1PenaltyUnit::new(1.0);
        let weight: NodeRef = NodeRef::new(WeightNode::new(4, 2, 1.0));

        let old_weights = weight.borrow_mut().get_data();
        println!("Weights: {:?}", old_weights);

        unit.add_parameter_input(&weight);

        let output_ref = unit.get_output_ref();
        output_ref.borrow_mut().apply_operation();
        output_ref.borrow_mut().add_gradient(&DataContainer::one());
        output_ref.borrow_mut().apply_jacobian();

        let new_weights = weight.borrow_mut().get_data();
        println!("New weights: {:?}", new_weights);
    }
}
