// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::unit::{types::input_unit::InputUnit, UnitContainer};

#[derive(Serialize, Deserialize)]
pub struct InputParams {
    pub input_size: Vec<usize>,
}

impl InputParams {
    pub fn from_unit<'a>(unit: &UnitContainer<'a, InputUnit<'a>>) -> InputParams {
        InputParams {
            input_size: unit.borrow().get_input_size().to_vec(),
        }
    }
}
