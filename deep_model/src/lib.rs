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
        data::Data,
        unit::{
            types::{input_unit::InputUnit, linear_unit::LinearUnit},
            Unit, UnitContainer,
        },
    };

    #[test]
    fn node_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(3));
        let hidden: UnitContainer<LinearUnit> = UnitContainer::new(LinearUnit::new("relu", 3, 2));

        let input_arr: Array1<f32> = arr1(&[1.0, 0.1, 1.0]);
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

        let end_data: Data = output.borrow_mut().get_data();

        println!("{:?}", end_data);
    }

    #[test]
    fn backprop_test() {
        todo!()
    }
}
