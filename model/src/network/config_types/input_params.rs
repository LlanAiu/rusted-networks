// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::unit::{types::input_unit::InputUnit, UnitContainer};

#[derive(Serialize, Deserialize)]
pub struct InputParams {
    input_size: Vec<usize>,
    keep_probability: f32,
}

impl InputParams {
    pub fn new(input_size: Vec<usize>, keep_probability: f32) -> InputParams {
        InputParams {
            input_size,
            keep_probability,
        }
    }

    pub fn from_unit<'a>(unit: &UnitContainer<'a, InputUnit<'a>>) -> InputParams {
        InputParams {
            input_size: unit.borrow().get_input_size().to_vec(),
            keep_probability: unit.borrow().get_mask_type().probability(),
        }
    }

    pub fn get_input_size(&self) -> &Vec<usize> {
        &self.input_size
    }

    pub fn get_keep_probability(&self) -> f32 {
        self.keep_probability
    }
}
