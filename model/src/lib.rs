// builtin

// external

// internal
pub mod data;
pub mod network;
pub mod node;
pub mod regularization;
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
                binary_classifier::BinaryClassifierNetwork,
                simple_classifier::SimpleClassifierNetwork,
                simple_regressor::SimpleRegressorNetwork,
            },
            Network,
        },
        regularization::penalty::{l2_penalty::builder::L2PenaltyBuilder, PenaltyConfig},
        unit::{
            types::{input_unit::InputUnit, loss_unit::LossUnit, softmax_unit::SoftmaxUnit},
            Unit, UnitContainer,
        },
    };

    #[test]
    fn loss_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(vec![3]));
        let hidden: UnitContainer<SoftmaxUnit> =
            UnitContainer::new(SoftmaxUnit::new("relu", 3, 2, 0.001));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(vec![2], "base_cross_entropy"));

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
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(vec![3]));
        let hidden: UnitContainer<SoftmaxUnit> =
            UnitContainer::new(SoftmaxUnit::new("relu", 3, 2, 0.001));
        let loss: UnitContainer<LossUnit> =
            UnitContainer::new(LossUnit::new(vec![2], "base_cross_entropy"));

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
        let classifier: SimpleClassifierNetwork =
            SimpleClassifierNetwork::new(vec![3], vec![2], vec![5], 0.001);

        let input_arr1: Array1<f32> = arr1(&[0.4, 0.1, 1.0]);
        let input = DataContainer::Inference(Data::VectorF32(input_arr1));

        let output = classifier.predict(input);
        println!("{:?}", output);
    }

    #[test]
    fn binary_classification_test() {
        let classifier: BinaryClassifierNetwork =
            BinaryClassifierNetwork::new(vec![1], vec![2], 0.05);

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let before_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let before_output = classifier.predict(before_data);
        println!("Before: {:?}", before_output);

        for _i in 0..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 0..8 {
                let rand = random_range(0.0..1.0);
                if rand < 0.5 {
                    let x: f32 = random_range(-1.0..-0.5);

                    inputs.push(Data::VectorF32(arr1(&[x])));
                    responses.push(Data::VectorF32(arr1(&[0.0])));
                } else {
                    let x: f32 = random_range(0.5..1.0);

                    inputs.push(Data::VectorF32(arr1(&[x])));
                    responses.push(Data::VectorF32(arr1(&[1.0])));
                }
            }

            let input = DataContainer::Batch(inputs);
            let response = DataContainer::Batch(responses);

            classifier.train(input, response);
        }

        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("After: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[0.6]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("After 2: {:?}", after_output2);

        classifier
            .save_to_file("test/binary_classifier_test.json")
            .expect("Save failed");
    }

    #[test]
    fn classification_test() {
        let classifier: SimpleClassifierNetwork =
            SimpleClassifierNetwork::new(vec![1], vec![2], vec![2], 0.05);

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let before_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let before_output = classifier.predict(before_data);
        println!("Before: {:?}", before_output);

        for _i in 0..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 0..8 {
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

        let test_arr2: Array1<f32> = arr1(&[0.6]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("After 2: {:?}", after_output2);

        classifier
            .save_to_file("test/classifier_test.json")
            .expect("Save failed");
    }

    #[test]
    fn binary_classifier_load_test() {
        let classifier: BinaryClassifierNetwork =
            BinaryClassifierNetwork::load_from_file("test/binary_classifier_test.json");

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[0.6]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);
    }

    #[test]
    fn classifier_load_test() {
        let classifier: SimpleClassifierNetwork =
            SimpleClassifierNetwork::load_from_file("test/classifier_test.json");

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[0.6]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = classifier.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);
    }

    #[test]
    fn regressor_load_test() {
        let regressor: SimpleRegressorNetwork =
            SimpleRegressorNetwork::load_from_file("test/regressor_test.json");

        let test_arr: Array1<f32> = arr1(&[2.0]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = regressor.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[3.0]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = regressor.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);
    }

    #[test]
    fn regressor_regularization_test() {
        let config: PenaltyConfig = PenaltyConfig::new(L2PenaltyBuilder::new(0.2));

        let regressor: SimpleRegressorNetwork =
            SimpleRegressorNetwork::new(vec![1], vec![1], vec![6, 3], 0.005, config, false);

        for _i in 0..200 {
            let mut inputs = Vec::new();
            let mut responses = Vec::new();

            for _j in 0..8 {
                let x: f32 = random_range(1.0..4.0);

                inputs.push(Data::VectorF32(arr1(&[x])));
                responses.push(Data::VectorF32(arr1(&[x * x])));
            }

            let input = DataContainer::Batch(inputs);
            let response = DataContainer::Batch(responses);

            regressor.train(input, response);
        }

        let test_arr: Array1<f32> = arr1(&[2.0]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = regressor.predict(after_data);
        println!("Loaded output 1: {:?}", after_output);

        let test_arr2: Array1<f32> = arr1(&[3.0]);
        let after_data2 = DataContainer::Inference(Data::VectorF32(test_arr2.clone()));
        let after_output2 = regressor.predict(after_data2);
        println!("Loaded output 2: {:?}", after_output2);

        regressor
            .save_to_file("test/regressor_test.json")
            .expect("Save Failed");
    }
}
