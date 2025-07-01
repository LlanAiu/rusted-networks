// builtin
use std::{mem::take, rc::Rc};

// external
use ndarray::{Array1, Array2, Axis};

// internal
use crate::node::{Data, Node, NodeRef};

pub struct MultiplyNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
}

impl<'a> MultiplyNode<'a> {
    pub fn new() -> MultiplyNode<'a> {
        return MultiplyNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
        };
    }

    fn update_data(&self, matrix: Array2<f32>) -> Data {
        let mut product: Array2<f32> = matrix;

        for i in 1..self.inputs.len() {
            let mut node_ref = self.inputs.get(i).unwrap().borrow_mut();
            let data = node_ref.get_data();

            match data {
                Data::MatrixF32(matrix) => {
                    product = product.dot(&matrix);
                }
                Data::VectorF32(vec) => {
                    product = product.dot(&vec.insert_axis(Axis(1)));
                }
                _ => {}
            };
        }

        let product_vec: Array1<f32> = product.remove_axis(Axis(1));
        Data::VectorF32(product_vec)
    }
}

impl<'a> Node<'a> for MultiplyNode<'a> {
    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        input.borrow_mut().add_output(this);
        self.inputs.push(Rc::clone(input));
    }

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.outputs.push(Rc::clone(output));
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        &self.inputs
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        &self.outputs
    }

    fn get_data(&mut self) -> Data {
        take(&mut self.data)
    }

    fn apply_operation(&mut self) {
        if self.inputs.len() == 0 {
            return;
        }

        for input in &self.inputs {
            input.borrow_mut().apply_operation();
        }

        let mut first_ref = self.inputs.get(0).unwrap().borrow_mut();
        let first_data = first_ref.get_data();

        match first_data {
            Data::MatrixF32(matrix) => {
                self.data = self.update_data(matrix);
            }
            Data::VectorF32(vec) => {
                let matrix = vec.insert_axis(Axis(1));
                self.data = self.update_data(matrix);
            }
            Data::None => {
                self.data = Data::None;
            }
        }
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }

    fn set_data(&mut self, data: Data) {
        panic!("[MATMUL] Unsupported Operation: Cannot set data of an operation node");
    }
}
