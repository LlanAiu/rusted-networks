// builtin

// external

use ndarray::arr1;

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    trainer::error::PredictionError,
};

pub trait SupervisedExample {
    fn get_response(&self) -> Data;

    fn get_input(&self) -> Data;

    fn get_test_error(&self, predicted: DataContainer) -> PredictionError;
}

pub struct QuadraticExample {
    input: f32,
    output: f32,
}

impl QuadraticExample {
    pub fn new(input: f32) -> QuadraticExample {
        QuadraticExample {
            input,
            output: input * input,
        }
    }
}

impl SupervisedExample for QuadraticExample {
    fn get_response(&self) -> Data {
        Data::VectorF32(arr1(&[self.output]))
    }

    fn get_input(&self) -> Data {
        Data::VectorF32(arr1(&[self.input]))
    }

    fn get_test_error(&self, predicted: DataContainer) -> PredictionError {
        match predicted {
            DataContainer::Inference(data) => {
                if let Data::VectorF32(vec) = data {
                    let mut loss = self.output - vec[0];
                    loss *= loss;
                    return PredictionError::Loss { loss };
                }
                panic!("Invalid data type input for test error");
            }
            _ => panic!("Invalid data type input for test error"),
        }
    }
}
