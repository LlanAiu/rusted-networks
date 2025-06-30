// builtin
use std::{mem::take, rc::Rc};

// external

// internal
use crate::node::{activation::activation_function::ActivationFunction, Data, Node, NodeRef};

pub struct ActivationNode<'a> {
    inputs: Vec<NodeRef<'a>>,
    outputs: Vec<NodeRef<'a>>,
    data: Data,
    function: ActivationFunction,
}

impl<'a> ActivationNode<'a> {
    pub fn new(function: ActivationFunction) -> ActivationNode<'a> {
        ActivationNode {
            inputs: Vec::new(),
            outputs: Vec::new(),
            data: Data::None,
            function,
        }
    }

    fn add_input(&mut self, this: NodeRef<'a>, input: NodeRef<'a>) {
        input.borrow_mut().add_output(Rc::clone(&this));

        if self.inputs.len() > 0 {
            self.inputs.clear();
            println!("[ACTIVATION] reassigning previously set input node");
        }

        self.inputs.push(input);
    }
}

impl<'a> Node<'a> for ActivationNode<'a> {
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
            println!("[ACTIVATION] Tried to apply operation on no inputs");
            return;
        }

        let mut input_ref = self.inputs.get(0).unwrap().borrow_mut();
        let mut data = input_ref.get_data();

        self.function.apply_all(&mut data);

        self.data = data;
    }

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
