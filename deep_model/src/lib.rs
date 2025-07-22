// builtin

// external

// internal
pub mod data;
pub mod network;
pub mod node;
pub mod unit;

#[cfg(test)]
mod tests {
    use std::vec;

    use approx::abs_diff_eq;
    use ndarray::{arr1, arr2, Array1, Array2};
    use rand::random_range;

    use crate::{
        data::{data_container::DataContainer, Data},
        network::{
            types::{
                simple_classifier::SimpleClassifierNetwork,
                simple_regressor::SimpleRegressorNetwork,
            },
            Network,
        },
        unit::{
            types::{input_unit::InputUnit, loss_unit::LossUnit, softmax_unit::SoftmaxUnit},
            Unit, UnitContainer,
        },
    };

    #[test]
    fn loss_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(&[3]));
        let hidden: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new("relu", 3, 2));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(&[2], "base_cross_entropy"));

        let input_arr: Array1<f32> = arr1(&[0.7, 0.1, 1.0]);
        input
            .borrow_mut()
            .set_input_data(DataContainer::Inference(Data::VectorF32(input_arr)));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        hidden
            .borrow_mut()
            .set_biases(DataContainer::Parameter(Data::VectorF32(biases_arr)));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        hidden
            .borrow_mut()
            .set_weights(DataContainer::Parameter(Data::MatrixF32(weight_arr)));

        let response_arr: Array1<f32> = arr1(&[0.9, 0.1]);
        loss.borrow()
            .set_expected_response(DataContainer::Inference(Data::VectorF32(response_arr)));

        hidden.add_input(&input);
        loss.add_input(&hidden);

        let loss_ref = loss.borrow();
        let output = loss_ref.get_output_node();

        output.borrow_mut().apply_operation();

        let end_data: DataContainer = output.borrow_mut().get_data();

        let expected: f32 = 0.4974879;

        match end_data {
            DataContainer::Inference(Data::ScalarF32(scalar)) => {
                assert!(
                    abs_diff_eq!(expected, scalar, epsilon = 1e-6),
                    "Output did not match expected value: got {:?}, expected {:?}",
                    scalar,
                    expected
                );
            }
            _ => panic!("Expected Inference(ScalarF32), got {:?}", end_data),
        };
    }

    #[test]
    fn batch_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(&[3]));
        let hidden: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new("relu", 3, 2));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(&[2], "base_cross_entropy"));

        let input_arr1: Array1<f32> = arr1(&[0.7, 0.1, 1.0]);
        let input_arr2: Array1<f32> = arr1(&[0.3, 0.8, 0.2]);
        input.borrow_mut().set_input_data(DataContainer::Batch(vec![
            Data::VectorF32(input_arr1),
            Data::VectorF32(input_arr2),
        ]));

        let biases_arr: Array1<f32> = arr1(&[0.5, 0.5]);
        hidden
            .borrow_mut()
            .set_biases(DataContainer::Parameter(Data::VectorF32(biases_arr)));

        let weight_arr: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        hidden
            .borrow_mut()
            .set_weights(DataContainer::Parameter(Data::MatrixF32(weight_arr)));

        let response_arr1: Array1<f32> = arr1(&[0.9, 0.1]);
        let response_arr2: Array1<f32> = arr1(&[0.3, 0.7]);
        loss.borrow()
            .set_expected_response(DataContainer::Batch(vec![
                Data::VectorF32(response_arr1),
                Data::VectorF32(response_arr2),
            ]));

        hidden.add_input(&input);
        loss.add_input(&hidden);

        let loss_ref = loss.borrow();
        let output = loss_ref.get_output_node();

        output.borrow_mut().apply_operation();

        let end_data: DataContainer = output.borrow_mut().get_data();

        println!("{:?}", end_data);
    }

    #[test]
    fn network_test() {
        let classifier: SimpleClassifierNetwork = SimpleClassifierNetwork::new(&[3], &[2], vec![5]);

        let input_arr1: Array1<f32> = arr1(&[0.4, 0.1, 1.0]);
        let input = DataContainer::Inference(Data::VectorF32(input_arr1));

        let layers = classifier.get_hidden_layers();
        println!("Hidden layers: {}", layers);

        let output = classifier.predict(input);
        println!("{:?}", output);
    }

    #[test]
    fn backprop_test() {
        let classifier: SimpleClassifierNetwork = SimpleClassifierNetwork::new(&[3], &[2], vec![5]);

        let input_arr1: Array1<f32> = arr1(&[0.4, 0.1, 1.0]);
        let input_arr2: Array1<f32> = arr1(&[0.3, 0.8, 0.2]);
        let input = DataContainer::Batch(vec![
            Data::VectorF32(input_arr1),
            Data::VectorF32(input_arr2),
        ]);

        let response_arr1: Array1<f32> = arr1(&[0.1795, 0.8205]);
        let response_arr2: Array1<f32> = arr1(&[0.5474, 0.4526]);
        let response = DataContainer::Batch(vec![
            Data::VectorF32(response_arr1),
            Data::VectorF32(response_arr2),
        ]);

        classifier.train(input, response);
    }

    #[test]
    fn training_test() {
        let classifier: SimpleClassifierNetwork = SimpleClassifierNetwork::new(&[1], &[2], vec![3]);

        let test_arr: Array1<f32> = arr1(&[-0.8]);
        let before_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let before_output = classifier.predict(before_data);
        println!("Before: {:?}", before_output);

        for _i in 1..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 1..8 {
                let rand = random_range(0.0..1.0);
                if rand < 0.5 {
                    let x: f32 = random_range(-1.0..-0.5);

                    inputs.push(Data::VectorF32(arr1(&[x])));
                    responses.push(Data::VectorF32(arr1(&[1.0, 0.0])));
                } else {
                    let x: f32 = random_range(0.5..1.0);

                    inputs.push(Data::VectorF32(arr1(&[x])));
                    responses.push(Data::VectorF32(arr1(&[0.0, 1.0])));
                }
            }

            let input = DataContainer::Batch(inputs);
            let response = DataContainer::Batch(responses);

            classifier.train(input, response);
        }

        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("After: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[0.8]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("After 2: {:?}", after_output2);
    }

    #[test]
    fn quadratic_test() {
        let classifier: SimpleRegressorNetwork = SimpleRegressorNetwork::new(&[1], &[1], vec![4]);

        let test_arr: Array1<f32> = arr1(&[2.0]);
        let before_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let before_output = classifier.predict(before_data);
        println!("Before: {:?}", before_output);

        for _i in 1..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 1..8 {
                let x: f32 = random_range(1.0..4.0);

                inputs.push(Data::VectorF32(arr1(&[x])));
                responses.push(Data::VectorF32(arr1(&[x * x])));
            }

            let input = DataContainer::Batch(inputs);
            let response = DataContainer::Batch(responses);

            classifier.train(input, response);
        }

        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("After: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[3.0]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("After 2: {:?}", after_output2);
    }
}
