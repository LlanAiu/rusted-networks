// builtin

// external

// internal
use crate::{import_csv::load_data_from_csv, types::HandwrittenExample};
use model::{
    network::types::classifier::ClassifierNetwork,
    optimization::momentum::DescentType,
    regularization::penalty::PenaltyConfig,
    trainer::{trainer_params::TrainerConfig, SupervisedTrainer},
};
pub mod import_csv;
pub mod types;

pub fn train_dataset() {
    let penalty_config: PenaltyConfig = PenaltyConfig::none();
    let classifier: ClassifierNetwork = ClassifierNetwork::new(
        vec![784],
        vec![10],
        vec![50],
        0.01,
        penalty_config,
        false,
        DescentType::nesterov(0.95),
    );

    let train: Vec<HandwrittenExample> =
        load_data_from_csv("../data/mnist_train.csv", 0, 250).expect("Failed to read data");
    let test: Vec<HandwrittenExample> =
        load_data_from_csv("../data/mnist_test.csv", 0, 50).expect("Failed to read data");

    let config: TrainerConfig<HandwrittenExample> = TrainerConfig::new(5, 4, train, test);

    let trainer: SupervisedTrainer<ClassifierNetwork, HandwrittenExample> =
        SupervisedTrainer::new(classifier, config);

    trainer.train("test/mnist_build_test.json");
}

#[cfg(test)]
mod tests {
    use model::{
        data::{data_container::DataContainer, Data},
        network::{types::classifier::ClassifierNetwork, Network},
        optimization::momentum::DescentType,
        regularization::penalty::PenaltyConfig,
        trainer::{examples::SupervisedExample, trainer_params::TrainerConfig, SupervisedTrainer},
    };

    use crate::{import_csv::load_data_from_csv, types::HandwrittenExample};

    #[test]
    fn tiny_dataset_init() {
        let penalty_config: PenaltyConfig = PenaltyConfig::none();
        let classifier: ClassifierNetwork = ClassifierNetwork::new(
            vec![784],
            vec![10],
            vec![50],
            0.01,
            penalty_config,
            false,
            DescentType::Base,
        );

        let data =
            load_data_from_csv("../data/mnist_train.csv", 0, 2).expect("Failed to read data");

        for _i in 0..60 {
            let mut inputs: Vec<Data> = Vec::new();
            let mut responses: Vec<Data> = Vec::new();

            for label in &data {
                inputs.push(label.get_input());
                responses.push(label.get_response());
            }

            classifier.train(
                DataContainer::Batch(inputs),
                DataContainer::Batch(responses),
            );
        }

        let test_label_one = &data[0];
        let test_input_one = DataContainer::Inference(test_label_one.get_input());
        let output_one = classifier.predict(test_input_one);
        println!("Output (5): {:?}", output_one);

        let test_label_two = &data[1];
        let test_input_two = DataContainer::Inference(test_label_two.get_input());
        let output_two = classifier.predict(test_input_two);
        println!("Output (0): {:?}", output_two);

        classifier
            .save_to_file("test/mnist_classifier_mini.json")
            .expect("Save failed");
    }

    #[test]
    fn tiny_dataset_load() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/mnist_classifier_mini.json");

        let data =
            load_data_from_csv("../data/mnist_train.csv", 0, 2).expect("Failed to read data");

        let test_label_one = &data[0];
        let test_input_one = DataContainer::Inference(test_label_one.get_input());
        let output_one = classifier.predict(test_input_one);
        println!("Output (5): {:?}", output_one);

        let test_label_two = &data[1];
        let test_input_two = DataContainer::Inference(test_label_two.get_input());
        let output_two = classifier.predict(test_input_two);
        println!("Output (0): {:?}", output_two);
    }

    #[test]
    fn small_dataset_init() {
        let penalty_config: PenaltyConfig = PenaltyConfig::none();
        let classifier: ClassifierNetwork = ClassifierNetwork::new(
            vec![784],
            vec![10],
            vec![50],
            0.01,
            penalty_config,
            false,
            DescentType::nesterov(0.95),
        );

        let train: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_train.csv", 0, 100).expect("Failed to read data");
        let test: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_test.csv", 0, 20).expect("Failed to read data");

        let config: TrainerConfig<HandwrittenExample> = TrainerConfig::new(5, 4, train, test);

        let trainer: SupervisedTrainer<ClassifierNetwork, HandwrittenExample> =
            SupervisedTrainer::new(classifier, config);

        trainer.train("test/mnist_small.json");
    }

    #[test]
    fn small_dataset_load_train() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/mnist_small.json");

        let train: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_train.csv", 0, 100).expect("Failed to read data");
        let test: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_test.csv", 0, 20).expect("Failed to read data");

        let config: TrainerConfig<HandwrittenExample> = TrainerConfig::new(5, 4, train, test);

        let trainer: SupervisedTrainer<ClassifierNetwork, HandwrittenExample> =
            SupervisedTrainer::new(classifier, config);

        trainer.train("test/mnist_small.json");
    }

    #[test]
    fn small_dataset_load_test() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/mnist_small.json");

        let data = load_data_from_csv("../data/mnist_test.csv", 0, 2).expect("Failed to read data");

        let test_label_one = &data[0];
        let test_input_one = DataContainer::Inference(test_label_one.get_input());
        let output_one = classifier.predict(test_input_one);
        println!("Output (7): {:?}", output_one);

        let test_label_two = &data[1];
        let test_input_two = DataContainer::Inference(test_label_two.get_input());
        let output_two = classifier.predict(test_input_two);
        println!("Output (2): {:?}", output_two);
    }

    #[test]
    fn med_dataset_init() {
        let penalty_config: PenaltyConfig = PenaltyConfig::none();
        let classifier: ClassifierNetwork = ClassifierNetwork::new(
            vec![784],
            vec![10],
            vec![72],
            0.01,
            penalty_config,
            false,
            DescentType::nesterov(0.95),
        );

        let train: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_train.csv", 0, 1000).expect("Failed to read data");
        let test: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_test.csv", 0, 500).expect("Failed to read data");

        let config: TrainerConfig<HandwrittenExample> = TrainerConfig::new(3, 4, train, test);

        let trainer: SupervisedTrainer<ClassifierNetwork, HandwrittenExample> =
            SupervisedTrainer::new(classifier, config);

        trainer.train("test/mnist_med.json");
    }

    #[test]
    fn med_dataset_load_train() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/mnist_med.json");

        // JUST RAN -- change offset to 40000 before running again
        let train: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_train.csv", 35000, 5000)
                .expect("Failed to read data");
        let test: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_test.csv", 0, 1000).expect("Failed to read data");

        let config: TrainerConfig<HandwrittenExample> = TrainerConfig::new(2, 4, train, test);

        let trainer: SupervisedTrainer<ClassifierNetwork, HandwrittenExample> =
            SupervisedTrainer::new(classifier, config);

        trainer.train("test/mnist_med.json");
    }

    #[test]
    fn med_dataset_load_test() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/mnist_med.json");

        let data = load_data_from_csv("../data/mnist_test.csv", 0, 2).expect("Failed to read data");

        let test_label_one = &data[0];
        let test_input_one = DataContainer::Inference(test_label_one.get_input());
        let output_one = classifier.predict(test_input_one);
        println!("Output (7): {:?}", output_one);

        let test_label_two = &data[1];
        let test_input_two = DataContainer::Inference(test_label_two.get_input());
        let output_two = classifier.predict(test_input_two);
        println!("Output (2): {:?}", output_two);
    }

    #[test]
    fn full_test() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/mnist_med.json");

        let train: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_train.csv", 0, 1).expect("Failed to read data");
        let test: Vec<HandwrittenExample> =
            load_data_from_csv("../data/mnist_test.csv", 0, 10000).expect("Failed to read data");

        let config: TrainerConfig<HandwrittenExample> = TrainerConfig::new(1, 4, train, test);
        let trainer: SupervisedTrainer<ClassifierNetwork, HandwrittenExample> =
            SupervisedTrainer::new(classifier, config);

        let error = trainer.evaluate();

        println!("Total Accuracy: {:?}", error);
    }
}
