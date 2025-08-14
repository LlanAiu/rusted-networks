// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    trainer::error::PredictionError,
};

pub trait SupervisedExample {
    fn get_response(&self) -> Data;

    fn get_input(&self) -> Data;

    fn get_error(&self, predicted: DataContainer) -> PredictionError;
}
