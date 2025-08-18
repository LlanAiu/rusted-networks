// builtin
use model::{
    data::{data_container::DataContainer, Data},
    trainer::{error::PredictionError, examples::SupervisedExample},
};
use std::{error::Error, io::ErrorKind};
// external
use ndarray::{arr1, Array1};

// internal

const ROW_LENGTH: usize = 28 * 28 + 1;
const OUTPUT_LENGTH: usize = 10;

#[derive(Debug)]
pub struct HandwrittenExample {
    label: usize,
    data: Array1<f32>,
}

impl HandwrittenExample {
    pub fn new(csv_row: Vec<u16>) -> Result<HandwrittenExample, Box<dyn Error>> {
        if csv_row.len() == ROW_LENGTH {
            let intensity: Vec<f32> = csv_row
                .iter()
                .skip(1)
                .map(|f| f32::from(*f) / 255.0)
                .collect();

            return Ok(HandwrittenExample {
                label: usize::from(csv_row[0]),
                data: arr1(&intensity),
            });
        }
        Err(Box::new(std::io::Error::new(
            ErrorKind::InvalidData,
            "Input data didn't match the expected length",
        )))
    }
}

impl SupervisedExample for HandwrittenExample {
    fn get_response(&self) -> Data {
        let mut vec: Vec<f32> = vec![0.0; OUTPUT_LENGTH];
        vec[self.label] = 1.0;

        Data::VectorF32(arr1(&vec))
    }

    fn get_input(&self) -> Data {
        Data::VectorF32(self.data.clone())
    }

    fn get_test_error(&self, predicted: DataContainer) -> PredictionError {
        if let DataContainer::Inference(Data::VectorF32(pred)) = predicted {
            if pred.dim() == OUTPUT_LENGTH {
                let mut max: f32 = pred[0];
                let mut max_index: usize = 0;

                for i in 1..OUTPUT_LENGTH {
                    let val = pred[i];

                    if val > max {
                        max = val;
                        max_index = i;
                    }
                }

                if max_index == self.label {
                    return PredictionError::Misclassification {
                        incorrect: 0,
                        total: 1,
                    };
                }

                return PredictionError::Misclassification {
                    incorrect: 1,
                    total: 1,
                };
            }
            panic!(
                "[MNIST] Invalid output dimension, expected 10 but got {}",
                pred.dim()
            );
        }
        panic!("[MNIST] Invalid data format for test error processing");
    }
}
