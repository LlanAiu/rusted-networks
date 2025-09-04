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

    use ndarray::{arr1, Array1};

    use crate::{
        data::{data_container::DataContainer, Data},
        network::{types::classifier::ClassifierNetwork, Network},
        optimization::{learning_decay::LearningDecay, momentum::DescentType},
        regularization::penalty::PenaltyConfig,
    };

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
