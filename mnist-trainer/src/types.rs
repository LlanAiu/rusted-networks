// builtin
use std::{error::Error, io::ErrorKind};
// external
use ndarray::{arr1, Array1};

// internal

const ROW_LENGTH: usize = 28 * 28 + 1;

#[derive(Debug)]
pub struct LabelledData {
    label: usize,
    data: Array1<f32>,
}

impl LabelledData {
    pub fn new(csv_row: Vec<u16>) -> Result<LabelledData, Box<dyn Error>> {
        if csv_row.len() == ROW_LENGTH {
            let intensity: Vec<f32> = csv_row
                .iter()
                .skip(1)
                .map(|f| f32::from(*f) / 255.0)
                .collect();

            return Ok(LabelledData {
                label: usize::from(csv_row[0]),
                data: arr1(&intensity),
            });
        }
        Err(Box::new(std::io::Error::new(
            ErrorKind::InvalidData,
            "Input data didn't match the expected length",
        )))
    }

    pub fn get_data(&self) -> Array1<f32> {
        self.data.clone()
    }

    pub fn get_label(&self) -> Array1<f32> {
        let mut vec: Vec<f32> = vec![0.0; 10];
        vec[self.label] = 1.0;

        arr1(&vec)
    }
}
