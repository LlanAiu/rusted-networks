// builtin

// external
use ndarray::Array1;

// internal
use crate::node::{node_base::NodeBase, Data, Node, NodeRef};

pub struct AddNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> AddNode<'a> {
    pub fn new() -> AddNode<'a> {
        AddNode {
            base: NodeBase::new(),
        }
    }
}

impl<'a> Node<'a> for AddNode<'a> {
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

        if let Data::VectorF32(vec) = first_data {
            let mut sum: Array1<f32> = vec;

            for i in 1..inputs.len() {
                let mut node_ref = inputs[i].borrow_mut();
                let data_one = node_ref.get_data();

                if let Data::VectorF32(vec) = data_one {
                    sum += &vec;
                }
            }

            self.base.set_data(Data::VectorF32(sum));
        } else {
            self.base.set_data(Data::None);
        }
    }

    fn set_data(&mut self, _data: Data) {
        panic!("[ADD] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &Data) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();

        //TODO: CALCULATIONS (WRT node)
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
