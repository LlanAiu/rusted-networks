// builtin

// external

// internal
pub mod data;
pub mod network;
pub mod node;
pub mod unit;

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1, Array2};

    use crate::{
        data::{data_container::DataContainer, Data},
        unit::{
            types::{input_unit::InputUnit, loss_unit::LossUnit, softmax_unit::SoftmaxUnit},
            Unit, UnitContainer,
        },
    };

    #[test]
    fn node_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(3));
        let hidden: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new("relu", 3, 2));

        let input_arr: Array1<f32> = arr1(&[0.7, 0.1, 1.0]);
        input
            .borrow_mut()
            .set_input_data(Data::VectorF32(input_arr));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        hidden.borrow_mut().set_biases(Data::VectorF32(biases_arr));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        hidden.borrow_mut().set_weights(Data::MatrixF32(weight_arr));

        hidden.add_input(&input);

        let hidden_ref = hidden.borrow();
        let output = hidden_ref.get_output_node();

        output.borrow_mut().apply_operation();

        let end_data: DataContainer = output.borrow_mut().get_data();

        println!("{:?}", end_data);
    }

    #[test]
    fn loss_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(3));
        let hidden: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new("relu", 3, 2));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(2, "base_cross_entropy"));

        let input_arr: Array1<f32> = arr1(&[0.7, 0.1, 1.0]);
        input.borrow().set_input_data(Data::VectorF32(input_arr));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        hidden.borrow().set_biases(Data::VectorF32(biases_arr));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        hidden.borrow().set_weights(Data::MatrixF32(weight_arr));

        let response_arr: Array1<f32> = arr1(&[0.9, 0.1]);
        loss.borrow()
            .set_expected_response(Data::VectorF32(response_arr));

        hidden.add_input(&input);
        loss.add_input(&hidden);

        let loss_ref = loss.borrow();
        let output = loss_ref.get_output_node();

        output.borrow_mut().apply_operation();

        let end_data: DataContainer = output.borrow_mut().get_data();

        println!("{:?}", end_data);
    }

    #[test]
    fn backprop_test() {
        todo!()
    }
}
