// builtin

// external

// internal
use crate::data::{data_container::DataContainer, Data};

pub enum LearningDecayType {
    Exponential {
        initial_rate: f32,
        decay_rate: f32,
    },
    LinearSchedule {
        initial_rate: f32,
        end_rate: f32,
        decay_time: usize,
    },
}

impl LearningDecayType {
    pub fn get_initial_rate(&self) -> f32 {
        match self {
            LearningDecayType::Exponential { initial_rate, .. } => *initial_rate,
            LearningDecayType::LinearSchedule { initial_rate, .. } => *initial_rate,
        }
    }

    pub fn get_next(&self, current: f32, time_step: usize) -> f32 {
        match self {
            LearningDecayType::Exponential {
                ..,
                decay_rate,
            } => current * decay_rate,
            LearningDecayType::LinearSchedule {
                initial_rate,
                end_rate,
                decay_time,
            } => {
                initial_rate - (time_step / decay_time) * (initial_rate - end_rate)
            }
        }
    }
}

pub struct LearningDecay {
    learning_rate: f32,
    time_step: usize,
    decay_type: LearningDecayType,
}

impl LearningDecay {
    pub fn new(decay_type: LearningDecayType) -> LearningDecay {
        LearningDecay {
            learning_rate: decay_type.get_initial_rate(),
            time_step: 0,
            decay_type: decay_type,
        }
    }

    pub fn get_learning_rate(&self) -> DataContainer {
        DataContainer::Parameter(Data::ScalarF32(self.learning_rate))
    }

    pub fn next(&mut self) {
        self.learning_rate = self.decay_type.get_next(self.learning_rate, self.time_step);
        self.time_step += 1;
    }
}
