// builtin

// external

// internal
use crate::{
    data::data_container::DataContainer,
    network::Network,
    unit::{
        types::{input_unit::InputUnit, linear_unit::LinearUnit, loss_unit::LossUnit},
        Unit, UnitContainer, UnitRef,
    },
};

pub struct BinaryClassiferNetwork<'a> {
    input: UnitContainer<'a, InputUnit<'a>>,
    _hidden: Vec<UnitContainer<'a, LinearUnit<'a>>>,
    inference: UnitContainer<'a, LinearUnit<'a>>,
    loss: UnitContainer<'a, LossUnit<'a>>,
}

impl<'a> BinaryClassiferNetwork<'a> {
    pub fn new(input_size: &'a [usize], hidden_sizes: Vec<usize>) -> BinaryClassiferNetwork<'a> {
        if input_size.len() != 1 {
            panic!("[BINARY_CLASSIFIER] Invalid input / output dimensions for network type, expected 1 but got {}.", input_size.len());
        }

        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(input_size));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(&[1], "binary_cross_entropy"));
        let mut hidden: Vec<UnitContainer<LinearUnit>> = Vec::new();

        let mut prev_width = input_size[0];
        let mut prev_unit: UnitRef = input.get_ref();

        for i in 0..hidden_sizes.len() {
            let width = hidden_sizes
                .get(i)
                .expect("[BINARY_CLASSIFIER] Failed to get layer width");

            if *width == 0 {
                println!("[BINARY_CLASSIFIER] got layer with 0 width, skipping creation...");
                continue;
            }

            let hidden_unit: UnitContainer<LinearUnit> =
                UnitContainer::new(LinearUnit::new("relu", prev_width, *width));

            hidden_unit.add_input_ref(&prev_unit);

            prev_unit = hidden_unit.get_ref();
            prev_width = *width;

            hidden.push(hidden_unit);
        }

        let inference: UnitContainer<LinearUnit> =
            UnitContainer::new(LinearUnit::new("sigmoid", prev_width, 1));
        inference.add_input_ref(&prev_unit);

        loss.add_input(&inference);

        BinaryClassiferNetwork {
            input,
            _hidden: hidden,
            inference,
            loss,
        }
    }
}

impl<'a> BinaryClassiferNetwork<'a> {
    pub fn get_hidden_layers(&self) -> usize {
        self._hidden.len()
    }
}

impl Network for BinaryClassiferNetwork<'_> {
    fn predict(&self, input: DataContainer) -> DataContainer {
        self.input.borrow_mut().set_input_data(input);

        let inference_ref = self.inference.borrow();
        let inference_node = inference_ref.get_output_node();

        inference_node.borrow_mut().apply_operation();

        let output = inference_node.borrow_mut().get_data();

        output
    }

    fn train(&self, input: DataContainer, response: DataContainer) {
        self.input.borrow().set_input_data(input);
        self.loss.borrow().set_expected_response(response);

        let loss_ref = self.loss.borrow();
        let loss_node = loss_ref.get_output_node();

        loss_node.borrow_mut().apply_operation();

        loss_node.borrow_mut().add_gradient(&DataContainer::one());
        loss_node.borrow_mut().apply_jacobian();
    }
}
