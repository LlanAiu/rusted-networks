// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    node::{node_base::NodeBase, Node, NodeRef, NodeType},
};

pub struct SquareNode<'a> {
    base: NodeBase<'a>,
}

impl<'a> SquareNode<'a> {
    pub fn new() -> SquareNode<'a> {
        SquareNode {
            base: NodeBase::new(),
        }
    }
}

impl<'a> Node<'a> for SquareNode<'a> {
    fn get_type(&self) -> NodeType {
        NodeType::Operation
    }

    fn add_input(&mut self, this: &NodeRef<'a>, input: &NodeRef<'a>) {
        if self.base.get_inputs().len() == 0 {
            self.base.add_input(this, input);
        } else {
            println!("[SQUARE] Node's maximum input capacity reached (1). Skipping assignment, consider using an extra node instead.");
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

    fn set_data(&mut self, _data: DataContainer) {
        println!("[SQUARE] Unsupported Operation: Cannot set data of an operation node");
    }

    fn get_data(&mut self) -> DataContainer {
        self.base.get_data()
    }

    fn apply_operation(&mut self) {
        let inputs = self.base.get_inputs();

        for input in inputs {
            input.borrow_mut().apply_operation();
        }

        let input_ref = inputs.get(0).unwrap();
        let input_data = input_ref.borrow_mut().get_data();

        let res = input_data.apply_function_ref(|data| data.times(data));

        self.base.set_data(res);
    }

    fn add_gradient(&mut self, grad: &DataContainer) {
        self.base.increment_grad_count();
        self.base.add_to_gradient(grad);
    }

    fn apply_jacobian(&mut self) {
        self.base.reset_grad_count();
        if self.base.get_inputs().len() == 0 {
            self.base.reset_gradient();
            return;
        }

        let scale = DataContainer::Parameter(Data::ScalarF32(2.0));
        let input = self.base.get_inputs().get(0).unwrap();

        let grad = input.borrow_mut().get_data().times(&scale);
        let prev_grad = self.base.get_gradient();

        let update = grad.times(&prev_grad);
        for node in self.base.get_inputs() {
            node.borrow_mut().add_gradient(&update);

            if node.borrow().should_process_backprop() {
                node.borrow_mut().apply_jacobian();
            }
        }

        self.base.reset_gradient();
    }

    fn should_process_backprop(&self) -> bool {
        self.base.should_process_backprop()
    }
}
