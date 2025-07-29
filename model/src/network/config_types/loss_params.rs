// builtin

// external
use serde::{Deserialize, Serialize};

// internal
use crate::unit::{types::loss_unit::LossUnit, UnitContainer};

#[derive(Serialize, Deserialize)]
pub struct LossParams {
    pub loss_type: String,
    pub output_size: Vec<usize>,
}

impl LossParams {
    pub fn from_unit<'a>(unit: &UnitContainer<'a, LossUnit<'a>>) -> LossParams {
        let loss_type = unit.borrow().get_loss_type().to_string();
        let output_size = unit.borrow().get_output_size().to_vec();

        LossParams {
            loss_type,
            output_size,
        }
    }
}
