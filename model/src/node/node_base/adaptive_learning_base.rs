// builtin

// external

// internal
use crate::{
    data::{data_container::DataContainer, Data},
    optimization::learning_decay::{LearningDecayType, LearningRateParams},
};

pub struct NodeLearningDecay {
    learning_rate: DataContainer,
    decay_type: LearningDecayType,
    time_step: usize,
    matches_dim: bool,
    is_adaptive: bool,
}

impl NodeLearningDecay {
    pub fn new(decay_type: LearningDecayType) -> NodeLearningDecay {
        NodeLearningDecay {
            learning_rate: decay_type.get_initial_rate(),
            is_adaptive: decay_type.is_adaptive(),
            time_step: decay_type.get_initial_timestep(),
            decay_type: decay_type,
            matches_dim: false,
        }
    }

    pub fn scale_update(&self, update: &mut DataContainer) {
        if self.is_adaptive {
            self.decay_type.scale_adaptive(update, &self.learning_rate);
        } else {
            update.times_assign(&self.learning_rate);
        }
    }

    // Assumes gradient has already been averaged (not a batch)
    pub fn update_learning_rate(&mut self, gradient: &DataContainer) {
        self.time_step += 1;

        if self.is_adaptive {
            if !self.matches_dim {
                self.learning_rate = DataContainer::zero_dim(gradient.dim().1);
                self.matches_dim = true;
            }
            self.decay_type
                .update_adaptive(&mut self.learning_rate, gradient, self.time_step);
        } else {
            self.decay_type
                .update_global(&mut self.learning_rate, self.time_step);
        }
    }

    pub fn get_learning_rate_save(&self) -> LearningRateParams {
        if let DataContainer::Parameter(data) = &self.learning_rate {
            return match data {
                Data::VectorF32(vec) => LearningRateParams::new(vec.to_vec()),
                Data::MatrixF32(matrix) => LearningRateParams::new(matrix.flatten().to_vec()),
                _ => LearningRateParams::null(),
            };
        }

        LearningRateParams::null()
    }

    pub fn set_learning_rate(&mut self, learning_rate: DataContainer) {
        self.learning_rate = learning_rate;
    }
}
