// builtin

// external
use ndarray::{ArrayView2, Axis};

// internal
use crate::data::Data;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct MatrixMultiplyNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> MatrixMultiplyNode<'a> {
    pub fn new() -> MatrixMultiplyNode<'a> {
        MatrixMultiplyNode {
            base: NodeBase::new(),
        }
    }

    fn propogate_gradient_of(&self, node_index: usize, other_index: usize) {
        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        let grad_data = self.base.get_gradient();
        let prev_gradient: ArrayView2<f32> = match grad_data {
            Data::VectorF32(vec) => vec.view().insert_axis(Axis(1)),
            Data::MatrixF32(matrix) => matrix.view(),
            _ => {
                let variant_name = grad_data.variant_name();
                panic!("[MATMUL] Invalid previous gradient, was {variant_name}");
            }
        };

        let mut node_ref = inputs.get(node_index).unwrap().borrow_mut();
        let mut other_ref = inputs.get(other_index).unwrap().borrow_mut();

        match other_ref.get_data() {
            Data::VectorF32(vec) => {
                let matrix = vec.insert_axis(Axis(1));
                let grad = prev_gradient.dot(&matrix.t());
                node_ref.add_gradient(&Data::MatrixF32(grad));
            }
            Data::MatrixF32(matrix) => {
                let grad = matrix.t().dot(&prev_gradient);
                node_ref.add_gradient(&Data::VectorF32(grad.remove_axis(Axis(1))));
            }
            _ => {
                let variant_name = grad_data.variant_name();
                panic!("[MATMUL] Invalid input data type for backprop, was {variant_name}");
            }
        }

        if node_ref.should_process_backprop() {
            node_ref.apply_jacobian();
        }
    }
}

impl<'a> Node<'a> for MatrixMultiplyNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() < 2 {
            self.base.add_input(this, input);
        } else {
            println!("[MATMUL] Node's maximum input capacity (2) reached. Skipping assignment, consider using an extra node instead.");
        }
    }

    fn add_output(&mut self, output: &NodeRef<'a>) {
        self.base.add_output(output);
    }

    fn get_inputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_inputs()
    }

    fn get_outputs(&self) -> &Vec<NodeRef<'a>> {
        self.base.get_outputs()
    }

    fn get_data(&mut self) -> Data {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        if self.get_inputs().len() != 2 {
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut first_ref = inputs.get(0).unwrap().borrow_mut();
        let first_data = first_ref.get_data();

        let mut second_ref = inputs.get(1).unwrap().borrow_mut();
        let second_data = second_ref.get_data();

        let res: Data = first_data.dot(&second_data);

        self.base.set_data(res);
    }

    fn set_data(&mut self, _data: Data) {
        println!("[MATMUL] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &Data) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        self.propogate_gradient_of(0, 1);
        self.propogate_gradient_of(1, 0);
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
