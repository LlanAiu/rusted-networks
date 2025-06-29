// builtin
use std::rc::Rc;

// external
use ndarray::{Array1, ArrayView2, Axis};

// internal
use crate::node::{Data, Node, NodeRef};

pub struct MultiplyNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    weight_node: Option<NodeRef<'a>>,
    input_node: Option<NodeRef<'a>>,
    data: Data,
}

impl<'a> MultiplyNode<'a> {
    pub fn new(output_dim: usize) -> MultiplyNode<'a> {
        return MultiplyNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            weight_node: None,
            input_node: None,
            data: Data::VectorF32(Array1::zeros(output_dim)),
        };
    }

    fn add_weight_input(&mut self, weight: NodeRef<'a>) {
        self.inputs.push(weight);
        self.update_inputs();
    }

    fn add_vector_input(&mut self, input: NodeRef<'a>) {
        self.inputs.push(input);
        self.update_inputs();
    }

    fn update_inputs(&mut self) {
        self.inputs.clear();
        if let Option::Some(node) = &self.input_node {
            self.inputs.push(Rc::clone(node));
        }
        if let Option::Some(node) = &self.weight_node {
            self.inputs.push(Rc::clone(node));
        }
    }
}

impl<'a> Node<'a> for MultiplyNode<'a> {
    fn add_output(&mut self, output: NodeRef<'a>) {
        self.outputs.push(Rc::clone(&output));
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    fn get_data(&self) -> &Data {
        &self.data
    }

    fn apply_operation(&mut self) {
        if self.inputs.len() != 2 {
            panic!(
                "Improper multiplicative node input number: expected 2 but got {}",
                self.inputs.len()
            );
        }

        if let Option::Some(weight_ref) = self.weight_node.clone() {
            if let Option::Some(input_ref) = self.input_node.clone() {
                let weight_data_ref = weight_ref.borrow();
                let weight_data: &Data = weight_data_ref.get_data();

                let input_data_ref = input_ref.borrow();
                let input_data: &Data = input_data_ref.get_data();

                if let Data::MatrixF32(weights) = weight_data {
                    if let Data::VectorF32(input) = input_data {
                        let input_matrix: ArrayView2<f32> = input.view().insert_axis(Axis(1));
                        let res: Array1<f32> = weights.dot(&input_matrix).remove_axis(Axis(1));

                        self.data = Data::VectorF32(res);
                        return;
                    }
                }
                panic!("[MATMUL] Invalid weight and/or input data structure");
            }
        }

        panic!("[MATMUL] Invalid input node references");
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
