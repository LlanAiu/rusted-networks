// builtin

// external

use crate::data::data_container::DataContainer;
// internal
use crate::node::loss::loss_function::LossFunction;
use crate::node::NodeType;
use crate::node::{node_base::NodeBase, Node, NodeRef};

pub struct LossNode<'a> {
    base: NodeBase<'a>,
    function: LossFunction,
}

impl<'a> LossNode<'a> {
    pub fn new(function_name: &str) -> LossNode<'a> {
        LossNode {
            base: NodeBase::new(),
            function: LossFunction::new(function_name),
        }
    }
}

impl<'a> Node<'a> for LossNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        match input.get_type() {
            NodeType::Operation | NodeType::ExpectedResponse => {
                let inputs = self.base.get_inputs().len();
                match inputs {
                    0 => {
                        self.base.add_input(this, input);
                    }
                    1 => {
                        let node = self.base.get_inputs().get(0).unwrap();
                        if node.get_type() != input.get_type() {
                            self.base.add_input(this, input);
                        } else {
                            println!("[LOSS] Node's maximum input capacity for type {} reached (1). Skipping assignment.", input.get_type());
                        }
                    }
                    _ => {
                        println!("[LOSS] Node's maximum input capacity reached (2). Skipping assignment.");
                    }
                }
            }
            _ => {
                println!(
                    "[LOSS] Tried to assign unsupported input type to loss node: {}",
                    input.get_type()
                );
            }
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
        if self.get_inputs().len() != 2 {
            println!(
                "[LOSS] Expected 2 Inputs but got {}, terminating feedforward operation",
                self.get_inputs().len()
            );
            return;
        }

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        for input in &inputs {
            input.borrow_mut().apply_operation();
        }

        let first_ref = inputs.get(0).unwrap();
        let first_data = first_ref.borrow_mut().get_data();

        let second_ref = inputs.get(1).unwrap();
        let second_data = second_ref.borrow_mut().get_data();

        if first_ref.get_type() == NodeType::ExpectedResponse {
            let data = self.function.apply(&first_data, &second_data);
            println!("Loss: {:?}", data);
            self.base.set_data(data);
        } else {
            let data = self.function.apply(&second_data, &first_data);
            println!("Loss: {:?}", data);
            self.base.set_data(data);
        }
    }

    fn set_data(&mut self, _data: DataContainer) {
        panic!("[LOSS] Unsupported Operation: Cannot set data of an operation node");
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
        let grad = self.base.get_gradient();

        let inputs: Vec<NodeRef<'a>> = self.get_inputs().iter().cloned().collect();

        let first_ref = inputs.get(0).unwrap();
        let first_data = first_ref.borrow_mut().get_data();

        let second_ref = inputs.get(1).unwrap();
        let second_data = second_ref.borrow_mut().get_data();

        if first_ref.get_type() == NodeType::Operation {
            let expected_grad = self.function.get_jacobian(&second_data, &first_data, true);
            second_ref
                .borrow_mut()
                .add_gradient(&expected_grad.times(grad));

            let actual_grad = self.function.get_jacobian(&second_data, &first_data, false);
            first_ref
                .borrow_mut()
                .add_gradient(&actual_grad.times(grad));
        } else {
            let expected_grad = self.function.get_jacobian(&first_data, &second_data, true);
            first_ref
                .borrow_mut()
                .add_gradient(&expected_grad.times(grad));

            let actual_grad = self.function.get_jacobian(&first_data, &second_data, false);
            second_ref
                .borrow_mut()
                .add_gradient(&actual_grad.times(grad));
        }

        for input in inputs {
            if input.borrow().should_process_backprop() {
                input.borrow_mut().apply_jacobian();
            }
        }
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
