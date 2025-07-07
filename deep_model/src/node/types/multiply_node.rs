// builtin

// external
use ndarray::{Array1, Array2, Axis};

// internal
use crate::node::{node_base::NodeBase, Data, Node, NodeRef};

pub struct MultiplyNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> MultiplyNode<'a> {
    pub fn new() -> MultiplyNode<'a> {
        MultiplyNode {
            base: NodeBase::new(),
        }
    }

    fn update_data(&self, matrix: Array2<f32>) -> Data {
        let mut product: Array2<f32> = matrix;

        for i in 1..self.base.get_inputs().len() {
            let mut node_ref = self.base.get_inputs().get(i).unwrap().borrow_mut();
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
        self.base.add_input(this, input);
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
        if self.get_inputs().len() == 0 {
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut first_ref = inputs.get(0).unwrap().borrow_mut();
        let first_data = first_ref.get_data();
        let mut res: Data = Data::None;

        match first_data {
            Data::MatrixF32(matrix) => {
                res = self.update_data(matrix);
            }
            Data::VectorF32(vec) => {
                let matrix = vec.insert_axis(Axis(1));
                res = self.update_data(matrix);
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

        //TODO: CALCULATIONS (WRT each node)
        for node in self.get_inputs() {
            node.borrow_mut().add_gradient(self.base.get_gradient());
            if node.borrow().should_process_backprop() {
                node.borrow_mut().apply_jacobian();
            }
        }
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
