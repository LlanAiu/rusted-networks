// builtin

// external

// internal
use crate::node::{node_base::NodeBase, Data, Node, NodeRef};

pub struct WeightNode<'a> {
    base: NodeBase<'a>,
    dim: (usize, usize),
}

impl<'a> WeightNode<'a> {
    pub fn new(input_size: usize, output_size: usize) -> WeightNode<'a> {
        WeightNode {
            base: NodeBase::new(),
            dim: (output_size, input_size),
        }
    }
}

impl<'a> Node<'a> for WeightNode<'a> {
    fn add_input(&mut self, _this: &NodeRef<'a>, _input: &NodeRef<'a>) {}

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.base.add_output(output);
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_outputs()
    }

    fn set_data(&mut self, input: Data) {
        if let Data::MatrixF32(matrix) = input {
            if matrix.dim() == self.dim {
                self.base.set_data(Data::MatrixF32(matrix));
                return;
            }
        }
        println!("[WEIGHT] type or dimension mismatch, skipping reassignment");
    }

    fn get_data(&mut self) -> Data {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {}

    fn get_jacobian(&self) -> Data {
        todo!()
    }
}
