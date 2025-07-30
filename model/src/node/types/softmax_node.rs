// builtin

use core::f32;

// external
use ndarray::Array2;

// internal
use crate::data::data_container::DataContainer;
use crate::data::Data;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct SoftmaxNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> SoftmaxNode<'a> {
    pub fn new() -> SoftmaxNode<'a> {
        SoftmaxNode {
            base: NodeBase::new(),
        }
    }

    fn epsilon() -> f32 {
        1e-7
    }

    fn softmax(data: Data) -> Data {
        if let Data::VectorF32(mut vec) = data {
            let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            vec.mapv_inplace(|f| f32::exp(f - max));

            let mut sum = vec.sum();
            if sum <= 0.0 {
                sum = SoftmaxNode::epsilon();
            }

            vec.mapv_inplace(|f| f / sum);

            Data::VectorF32(vec)
        } else {
            println!(
                "[SOFTMAX] Invalid data type. Expected Data::VectorF32 but got {}",
                data.variant_name()
            );
            Data::None
        }
    }

    fn softmax_jacobian(softmax: Data) -> Data {
        if let Data::VectorF32(vec) = softmax {
            let n = vec.len();

            let mut jacobian: Array2<f32> = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        jacobian[[i, j]] = vec[i] * (1.0 - vec[j]);
                    } else {
                        jacobian[[i, j]] = -vec[i] * vec[j];
                    }
                }
            }

            Data::MatrixF32(jacobian)
        } else {
            println!(
                "[SOFTMAX] Invalid data type. Expected Data::VectorF32 but got {}",
                softmax.variant_name()
            );
            Data::None
        }
    }
}

impl<'a> Node<'a> for SoftmaxNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() == 0 {
            self.base.add_input(this, input);
        } else {
            println!("[SOFTMAX] Node's maximum input capacity reached (1). Skipping assignment, consider using an extra node instead.");
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

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        if self.get_inputs().len() == 0 {
            println!("[SOFTMAX] Tried to apply operation on no inputs");
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let mut input_ref = inputs.get(0).unwrap().borrow_mut();

        let data = input_ref.get_data();

        let res = data.apply_function(SoftmaxNode::softmax);

        self.base.set_data(res);
    }

    fn set_data(&mut self, _data: DataContainer) {
        panic!("[SOFTMAX] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
        let data = self.get_data();

        if self.base.get_inputs().len() == 0 {
            return;
        }

        let node = self.base.get_inputs().get(0).unwrap();

        let jacobian = data.apply_function(SoftmaxNode::softmax_jacobian);
        let grad = jacobian.matmul(self.base.get_gradient());

        node.borrow_mut().add_gradient(&grad);

        if node.borrow().should_process_backprop() {
            node.borrow_mut().apply_jacobian();
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
