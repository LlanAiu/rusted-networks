// builtin

// external
use ndarray::{Array1, Array2};

// internal
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

    fn softmax(mut arr: Array1<f32>) -> Data {
        arr.mapv_inplace(|f| f32::exp(f));

        let sum = arr.sum();

        arr.mapv_inplace(|f| f / sum);

        Data::VectorF32(arr)
    }

    fn softmax_jacobian(softmax: Array1<f32>) -> Data {
        let n = softmax.len();

        let mut jacobian: Array2<f32> = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    jacobian[[i, j]] = softmax[i] * (1.0 - softmax[j]);
                } else {
                    jacobian[[i, j]] = -softmax[i] * softmax[j];
                }
            }
        }

        Data::MatrixF32(jacobian)
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

    fn get_data(&mut self) -> Data {
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

        let mut data = input_ref.get_data();

        if let Data::VectorF32(vec) = data {
            data = SoftmaxNode::softmax(vec);
            self.base.set_data(data);
        } else {
            println!(
                "[SOFTMAX] Invalid data type. Expected Data::VectorF32 but got {}",
                data.variant_name()
            );
            self.base.set_data(data);
        }
    }

    fn set_data(&mut self, _data: Data) {
        panic!("[SOFTMAX] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &Data) {
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

        if let Data::VectorF32(vec) = data {
            let grad = SoftmaxNode::softmax_jacobian(vec);

            node.borrow_mut()
                .add_gradient(&grad.matmul(self.base.get_gradient()));
        } else {
            println!(
                "[SOFTMAX] Invalid data type. Expected Data::VectorF32 but got {}",
                data.variant_name()
            );
            node.borrow_mut().add_gradient(self.base.get_gradient());
        }

        if node.borrow().should_process_backprop() {
            node.borrow_mut().apply_jacobian();
        }
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
