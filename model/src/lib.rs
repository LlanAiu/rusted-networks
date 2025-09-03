// builtin

// external

// internal
pub mod data;
pub mod network;
pub mod node;
pub mod optimization;
pub mod regularization;
pub mod trainer;
pub mod unit;

#[cfg(test)]
mod tests {
    use std::vec;

    use approx::abs_diff_eq;
    use ndarray::{arr1, arr2, Array1, Array2};

    use crate::{
        data::{data_container::DataContainer, Data},
        network::{types::classifier::ClassifierNetwork, Network},
        optimization::{
            learning_decay::{LearningDecay, LearningDecayType},
            momentum::DescentType,
        },
        regularization::penalty::PenaltyConfig,
        unit::{
            types::{input_unit::InputUnit, loss_unit::LossUnit, softmax_unit::SoftmaxUnit},
            Unit, UnitContainer,
        },
    };

    #[test]
    fn loss_test() {
        let input: UnitContainer<InputUnit> = UnitContainer::new(InputUnit::new(vec![3]));
        let hidden: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new(
            "relu",
            3,
            2,
            LearningDecayType::constant(0.001),
            DescentType::Base,
        ));
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
        let hidden: UnitContainer<SoftmaxUnit> = UnitContainer::new(SoftmaxUnit::new(
            "relu",
            3,
            2,
            LearningDecayType::constant(0.001),
            DescentType::Base,
        ));
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
        let penalty_config: PenaltyConfig = PenaltyConfig::none();

        let classifier: ClassifierNetwork = ClassifierNetwork::new(
            vec![3],
            vec![2],
            vec![5],
            LearningDecay::constant(0.001),
            penalty_config,
            false,
            DescentType::Base,
        );

        let input_arr1: Array1<f32> = arr1(&[0.4, 0.1, 1.0]);
        let input = DataContainer::Inference(Data::VectorF32(input_arr1));

        let output = classifier.predict(input);
        println!("{:?}", output);
    }
}
