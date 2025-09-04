// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    optimization::momentum::{DescentType, MomentumParams},
};

pub struct NodeMomentum {
    momentum: DataContainer,
    is_momentum_null: bool,
    descent_type: DescentType,
}

impl NodeMomentum {
    pub fn new(descent_type: DescentType) -> NodeMomentum {
        NodeMomentum {
            momentum: DataContainer::zero(),
            is_momentum_null: true,
            descent_type,
        }
    }

    pub fn alter_data(&self, data: &mut DataContainer) {
        if let DescentType::Nesterov { .. } = &self.descent_type {
            data.sum_assign(&self.momentum);
        }
    }

    pub fn reset_momentum(&mut self) {
        self.momentum = DataContainer::zero();
        self.is_momentum_null = true;
    }

    pub fn is_momentum_update(&self) -> bool {
        match self.descent_type {
            DescentType::Base => false,
            DescentType::Momentum { .. } => true,
            DescentType::Nesterov { .. } => true,
        }
    }

    pub fn get_momentum_update(&mut self, update: &DataContainer) -> &DataContainer {
        if self.is_momentum_null {
            self.momentum = update.clone();
            self.is_momentum_null = false;
            return &self.momentum;
        }

        match &self.descent_type {
            DescentType::Momentum { decay } => {
                self.momentum.apply_inplace(|f| *f *= decay);
                self.momentum.sum_assign(&update);
            }
            DescentType::Nesterov { decay } => {
                self.momentum.apply_inplace(|f| *f *= decay);
                self.momentum.sum_assign(&update);
            }
            _ => panic!("Tried to get invalid momentum update"),
        }

        &self.momentum
    }

    pub fn get_momentum_save(&self) -> MomentumParams {
        if self.is_momentum_null {
            return MomentumParams::new(Vec::new());
        }

        if let DataContainer::Parameter(data) = &self.momentum {
            return match data {
                Data::ScalarF32(scalar) => MomentumParams::new(vec![*scalar]),
                Data::VectorF32(vec) => MomentumParams::new(vec.to_vec()),
                Data::MatrixF32(matrix) => MomentumParams::new(matrix.flatten().to_vec()),
                Data::None => MomentumParams::new(Vec::new()),
            };
        }

        MomentumParams::new(Vec::new())
    }

    pub fn set_momentum(&mut self, momentum: DataContainer) {
        self.momentum = momentum;
    }
}
