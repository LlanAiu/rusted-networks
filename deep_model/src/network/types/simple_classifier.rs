// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::Network,
    unit::{
        types::{
            input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit,
            softmax_unit::SoftmaxUnit,
        },
        Unit, UnitContainer, UnitRef,
    },
};

pub struct SimpleClassifierNetwork<'a> {
    input: UnitContainer<'a, InputUnit<'a>>,
    hidden: Vec<UnitContainer<'a, LinearUnit<'a>>>,
    inference: UnitContainer<'a, SoftmaxUnit<'a>>,
    loss: UnitContainer<'a, LossUnit<'a>>,
}

// TODO: Find a more elegant way to handle input/output dimensions (either restrict to 1D or find a way to handle higher dims)
impl<'a> SimpleClassifierNetwork<'a> {
    pub fn new(
        input_size: &'a [usize],
        output_size: &'a [usize],
        hidden_sizes: Vec<usize>,
    ) -> SimpleClassifierNetwork<'a> {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(input_size));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(output_size, "base_cross_entropy"));
        let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();

        let mut prev_width = input_size[0];
        let mut prev_unit: UnitRef = input.get_ref();

        for i in 0..hidden_sizes.len() {
            let width = hidden_sizes.get(i).expect("Failed to get layer width");

            let hidden_unit: UnitContainer<LinearUnit> =
                UnitContainer::new(LinearUnit::new("relu", prev_width, *width));

            hidden_unit.add_input_ref(&prev_unit);

            prev_unit = hidden_unit.get_ref();
            prev_width = *width;

            hidden.push(hidden_unit);
        }

        let inference: UnitContainer<SoftmaxUnit> =
            UnitContainer::new(SoftmaxUnit::new("relu", prev_width, output_size[0]));
        inference.add_input_ref(&prev_unit);

        SimpleClassifierNetwork {
            input,
            hidden,
            inference,
            loss,
        }
    }
}

impl Network for SimpleClassifierNetwork<'_> {
    fn feedforward(&self, input: DataContainer) {
        self.input.borrow_mut().set_input_data(input);

        let inference_node = self.inference.borrow();
        let inference_ref = inference_node.get_output_node();

        inference_ref.borrow_mut().apply_operation();
    }

    fn backprop(&self) {
        let loss_unit = self.loss.borrow();
        let loss_node = loss_unit.get_output_node();

        loss_node.borrow_mut().apply_jacobian();
    }
}
