// builtin

// external

// internal
pub mod network;
pub mod node;
pub mod unit;

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use ndarray::{arr1, arr2, Array1, Array2};

    use crate::{
        node::Data,
        unit::{
            types::{input_unit::InputUnit, linear_unit::LinearUnit},
            UnitRef,
        },
    };

    #[test]
    fn node_test() {
        let input: UnitRef = Rc::new(RefCell::new(InputUnit::new(3)));
        let hidden_one: UnitRef = Rc::new(RefCell::new(LinearUnit::new("relu", 3, 2)));

        let input_arr: Array1<f32> = arr1(&[1.0, 0.1, 1.0]);
        input
            .borrow_mut()
            .set_input_data(Data::VectorF32(input_arr));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        hidden_one
            .borrow_mut()
            .set_biases(Data::VectorF32(biases_arr));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        hidden_one
            .borrow_mut()
            .set_weights(Data::MatrixF32(weight_arr));

        hidden_one.borrow_mut().add_input(&hidden_one, &input);

        let hidden_ref = hidden_one.borrow();
        let output = hidden_ref.get_output_node();

        output.borrow_mut().apply_operation();

        let end_data: Data = output.borrow_mut().get_data();

        println!("{:?}", end_data);
    }

    #[test]
    fn backprop_test() {
        let input: UnitRef = Rc::new(RefCell::new(InputUnit::new(3)));
        let hidden_one: UnitRef = Rc::new(RefCell::new(LinearUnit::new("relu", 3, 2)));

        let input_arr: Array1<f32> = arr1(&[1.0, 0.1, 1.0]);
        input
            .borrow_mut()
            .set_input_data(Data::VectorF32(input_arr));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        hidden_one
            .borrow_mut()
            .set_biases(Data::VectorF32(biases_arr));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        hidden_one
            .borrow_mut()
            .set_weights(Data::MatrixF32(weight_arr));

        hidden_one.borrow_mut().add_input(&hidden_one, &input);

        let hidden_ref = hidden_one.borrow();
        let output = hidden_ref.get_output_node();

        output.borrow_mut().apply_operation();

        let end_data: Data = output.borrow_mut().get_data();

        println!("{:?}", end_data);
    }
}
