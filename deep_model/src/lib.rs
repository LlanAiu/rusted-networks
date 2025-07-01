// builtin

// external

// internal
pub mod node;

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use ndarray::{arr1, arr2, Array1, Array2};

    use crate::node::{
        types::{
            activation_node::ActivationNode, add_node::AddNode, bias_node::BiasNode,
            input_node::InputNode, multiply_node::MultiplyNode, weight_node::WeightNode,
        },
        Data, NodeRef,
    };

    #[test]
    fn node_test() {
        let input: NodeRef = Rc::new(RefCell::new(InputNode::new(3)));
        let weights: NodeRef = Rc::new(RefCell::new(WeightNode::new(3, 2)));
        let biases: NodeRef = Rc::new(RefCell::new(BiasNode::new(2)));

        let multiply: NodeRef = Rc::new(RefCell::new(MultiplyNode::new()));
        let add: NodeRef = Rc::new(RefCell::new(AddNode::new()));
        let activation: NodeRef = Rc::new(RefCell::new(ActivationNode::new("relu")));

        multiply.borrow_mut().add_input(&multiply, &weights);
        multiply.borrow_mut().add_input(&multiply, &input);

        add.borrow_mut().add_input(&add, &multiply);
        add.borrow_mut().add_input(&add, &biases);

        activation.borrow_mut().add_input(&activation, &add);

        let input_arr: Array1<f32> = arr1(&[1.0, 0.1, 1.0]);
        input.borrow_mut().set_data(Data::VectorF32(input_arr));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        biases.borrow_mut().set_data(Data::VectorF32(biases_arr));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        weights.borrow_mut().set_data(Data::MatrixF32(weight_arr));

        activation.borrow_mut().apply_operation();

        let end_data: Data = activation.borrow_mut().get_data();

        println!("{:?}", end_data);
    }
}
