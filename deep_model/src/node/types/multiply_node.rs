// builtin

// external
use ndarray::{Array1, Array2, ArrayView2, Axis};

// internal
use crate::data::Data;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct MultiplyNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> MultiplyNode<'a> {
    pub fn new() -> MultiplyNode<'a> {
        MultiplyNode {
            base: NodeBase::new(),
        }
    }

    fn update_data(&self, matrix: Array2<f32>, other_data: Data) -> Data {
        let mut product: Array2<f32> = matrix;

        match other_data {
            Data::MatrixF32(matrix) => {
                product = product.dot(&matrix);
            }
            Data::VectorF32(vec) => {
                product = product.dot(&vec.insert_axis(Axis(1)));
            }
            _ => {}
        };

        let product_vec: Array1<f32> = product.remove_axis(Axis(1));
        Data::VectorF32(product_vec)
    }

    fn propogate_gradient_of(&self, node_index: usize, other_index: usize) {
        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        let prev_gradient: ArrayView2<f32> = match self.base.get_gradient() {
            Data::VectorF32(vec) => vec.view().insert_axis(Axis(1)),
            Data::MatrixF32(matrix) => matrix.view(),
            Data::None => {
                panic!("[MATMUL] Invalid previous gradient, was Data::None");
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
            Data::None => {}
        }

        if node_ref.should_process_backprop() {
            node_ref.apply_jacobian();
        }
    }
}

impl<'a> Node<'a> for MultiplyNode<'a> {
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
        let mut res: Data = Data::None;

        let mut second_ref = inputs.get(1).unwrap().borrow_mut();
        let second_data = second_ref.get_data();

        match first_data {
            Data::MatrixF32(matrix) => {
                res = self.update_data(matrix, second_data);
            }
            Data::VectorF32(vec) => {
                let matrix = vec.insert_axis(Axis(1));
                res = self.update_data(matrix, second_data);
            }
            _ => {}
        }

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
