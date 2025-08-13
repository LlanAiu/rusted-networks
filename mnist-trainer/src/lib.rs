// builtin

// external

// internal
pub mod import_csv;
pub mod types;

#[cfg(test)]
mod tests {
    use model::{
        data::{data_container::DataContainer, Data},
        network::{types::classifier::ClassifierNetwork, Network},
        regularization::penalty::PenaltyConfig,
    };
    use ndarray::{arr1, Array1};

    use crate::import_csv::load_data_from_csv;

    #[test]
    fn tiny_dataset_init() {
        let penalty_config: PenaltyConfig = PenaltyConfig::none();
        let classifier: ClassifierNetwork =
            ClassifierNetwork::new(vec![784], vec![10], vec![50], 0.01, penalty_config, false);

        let data = load_data_from_csv("../data/mnist_train.csv").expect("Failed to read data");

        for _i in 0..60 {
            let mut inputs: Vec<Data> = Vec::new();
            let mut responses: Vec<Data> = Vec::new();

            for label in &data {
                inputs.push(Data::VectorF32(label.get_data()));
                responses.push(Data::VectorF32(label.get_label()));
            }

            classifier.train(
                DataContainer::Batch(inputs),
                DataContainer::Batch(responses),
            );
        }

        let test_label_one = &data[0];
        let test_input_one = DataContainer::Inference(Data::VectorF32(test_label_one.get_data()));
        let output_one = classifier.predict(test_input_one);
        println!("Output (5): {:?}", output_one);

        let test_label_two = &data[1];
        let test_input_two = DataContainer::Inference(Data::VectorF32(test_label_two.get_data()));
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

        let data = load_data_from_csv("../data/mnist_train.csv").expect("Failed to read data");

        let test_label_one = &data[0];
        let test_input_one = DataContainer::Inference(Data::VectorF32(test_label_one.get_data()));
        let output_one = classifier.predict(test_input_one);
        println!("Output (5): {:?}", output_one);

        let test_label_two = &data[1];
        let test_input_two = DataContainer::Inference(Data::VectorF32(test_label_two.get_data()));
        let output_two = classifier.predict(test_input_two);
        println!("Output (0): {:?}", output_two);
    }

    #[test]
    fn can_access_model() {
        let classifier: ClassifierNetwork =
            ClassifierNetwork::load_from_file("test/classifier_test.json");

        let test_arr: Array1<f32> = arr1(&[-0.7]);
        let after_data = DataContainer::Inference(Data::VectorF32(test_arr.clone()));
        let after_output = classifier.predict(after_data);
        println!("Loaded output: {:?}", after_output);
    }
}
