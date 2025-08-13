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
    };
    use ndarray::{arr1, Array1};

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
