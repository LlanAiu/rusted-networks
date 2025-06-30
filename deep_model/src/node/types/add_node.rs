// builtin
use std::{mem::take, rc::Rc};

// external
use ndarray::Array1;

// internal
use crate::node::{Data, Node, NodeRef};

pub struct AddNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
}

impl<'a> AddNode<'a> {
    pub fn new() -> AddNode<'a> {
        AddNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
        }
    }

    fn add_input(&mut self, this: NodeRef<'a>, input: NodeRef<'a>) {
        input.borrow_mut().add_output(Rc::clone(&this));
        self.inputs.push(input);
    }
}

impl<'a> Node<'a> for AddNode<'a> {
    fn add_output(&mut self, output: NodeRef<'a>) {
        self.outputs.push(Rc::clone(&output));
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

        let mut first_ref = self.inputs.get(0).unwrap().borrow_mut();
        let first_data = first_ref.get_data();

        if let Data::VectorF32(vec) = first_data {
            let mut sum: Array1<f32> = vec;

            for i in 1..self.inputs.len() {
                let mut node_ref = self.inputs.get(i).unwrap().borrow_mut();
                let data_one = node_ref.get_data();

                if let Data::VectorF32(vec) = data_one {
                    sum += &vec;
                }
            }

            self.data = Data::VectorF32(sum);
        }
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
