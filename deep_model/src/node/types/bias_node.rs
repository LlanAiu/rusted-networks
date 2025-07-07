// builtin

// external

// internal
use crate::node::{node_base::NodeBase, Data, Node, NodeRef};

pub struct BiasNode<'a> {
    base: NodeBase<'a>,
    dim: usize,
    learning_rate: f32,
}

impl<'a> BiasNode<'a> {
    pub fn new(dim: usize, learning_rate: f32) -> BiasNode<'a> {
        BiasNode {
            base: NodeBase::new(),
            dim,
            learning_rate,
        }
    }
}

impl<'a> Node<'a> for BiasNode<'a> {
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
        if let Data::VectorF32(vec) = input {
            if vec.dim() == self.dim {
                self.base.set_data(Data::VectorF32(vec));
                return;
            }
        }
        println!("[BIAS] type or dimension mismatch, skipping reassignment");
    }

    fn get_data(&mut self) -> Data {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {}

    fn add_gradient(&mut self, grad: &Data) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
        self.base.process_gradient(self.learning_rate);
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
