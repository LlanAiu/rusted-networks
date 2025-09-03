// builtin

use std::mem::take;

// external
use serde::{Deserialize, Serialize};

// internal
use crate::data::{data_container::DataContainer, Data};

const DELTA: f32 = 1e-7;

#[derive(Serialize, Deserialize, Clone)]
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
    RMSProp {
        global_rate: f32,
        decay_rate: f32,
    },
    None {
        rate: f32,
    },
}

impl LearningDecayType {
    pub fn exponential(initial_rate: f32, decay_rate: f32) -> LearningDecayType {
        LearningDecayType::Exponential {
            initial_rate,
            decay_rate,
        }
    }

    pub fn linear_schedule(
        initial_rate: f32,
        end_rate: f32,
        decay_time: usize,
    ) -> LearningDecayType {
        LearningDecayType::LinearSchedule {
            initial_rate,
            end_rate,
            decay_time,
        }
    }

    pub fn constant(initial_rate: f32) -> LearningDecayType {
        LearningDecayType::None { rate: initial_rate }
    }

    pub fn rms_prop(global_rate: f32, decay_rate: f32) -> LearningDecayType {
        LearningDecayType::RMSProp {
            global_rate,
            decay_rate,
        }
    }

    pub fn get_initial_rate_f32(&self) -> f32 {
        match self {
            LearningDecayType::Exponential { initial_rate, .. } => *initial_rate,
            LearningDecayType::LinearSchedule { initial_rate, .. } => *initial_rate,
            LearningDecayType::RMSProp { global_rate, .. } => *global_rate,
            LearningDecayType::None { rate } => *rate,
        }
    }

    pub fn is_adaptive(&self) -> bool {
        match self {
            LearningDecayType::Exponential { .. } => false,
            LearningDecayType::LinearSchedule { .. } => false,
            LearningDecayType::RMSProp { .. } => true,
            LearningDecayType::None { .. } => false,
        }
    }

    pub fn get_initial_rate(&self) -> DataContainer {
        match self {
            LearningDecayType::Exponential { initial_rate, .. } => {
                DataContainer::Parameter(Data::ScalarF32(*initial_rate))
            }
            LearningDecayType::LinearSchedule { initial_rate, .. } => {
                DataContainer::Parameter(Data::ScalarF32(*initial_rate))
            }
            LearningDecayType::RMSProp { .. } => DataContainer::zero(),
            LearningDecayType::None { rate } => DataContainer::Parameter(Data::ScalarF32(*rate)),
        }
    }

    pub fn get_next(&self, current: f32, time_step: usize) -> f32 {
        match self {
            LearningDecayType::Exponential { decay_rate, .. } => current * decay_rate,
            LearningDecayType::LinearSchedule {
                initial_rate,
                end_rate,
                decay_time,
            } => {
                initial_rate
                    - ((time_step as f32) / (*decay_time as f32)) * (initial_rate - end_rate)
            }
            LearningDecayType::None { rate: initial_rate } => *initial_rate,
            LearningDecayType::RMSProp {
                global_rate,
                decay_rate,
            } => todo!(),
        }
    }

    pub fn update_global(&self, learning_rate: &mut DataContainer, time_step: usize) {
        match self {
            LearningDecayType::Exponential { decay_rate, .. } => {
                learning_rate.apply_inplace(|f| *f *= decay_rate);
            }
            LearningDecayType::LinearSchedule {
                initial_rate,
                end_rate,
                decay_time,
            } => {
                let percent = (time_step as f32) / (*decay_time as f32);
                let new_rate = initial_rate * (1.0 - percent) + percent * end_rate;
                learning_rate.apply_inplace(|f| *f = new_rate);
            }
            LearningDecayType::None { .. } => {}
            _ => {
                println!("Cannot compute global update on an adaptive learning rate configuration, try using .update_adaptive() instead");
            }
        }
    }

    pub fn update_adaptive(
        &self,
        learning_rate: &mut DataContainer,
        gradient: &DataContainer,
        _time_step: usize,
    ) {
        match self {
            LearningDecayType::RMSProp { decay_rate, .. } => {
                let learning_update: DataContainer =
                    gradient.apply_elementwise(|f| (1.0 - decay_rate) * f32::powi(f, 2));
                learning_rate.apply_inplace(|f| *f *= decay_rate);
                learning_rate.sum_assign(&learning_update);
            }
            _ => {
                println!("Cannot compute adaptive update on an global learning rate configuration, try using .update_global() instead");
            }
        }
    }

    pub fn scale_adaptive(&self, update: &mut DataContainer, learning_rate: &DataContainer) {
        match self {
            LearningDecayType::RMSProp { global_rate, .. } => {
                let scale =
                    learning_rate.apply_elementwise(|f| *global_rate / f32::sqrt(DELTA + f));

                update.times_assign(&scale);
            }
            _ => {
                panic!(
                    "Tried to adaptively scale update using a global learning rate configuration!"
                );
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LearningDecay {
    learning_rate: f32,
    time_step: usize,
    decay_type: LearningDecayType,
}

impl LearningDecay {
    pub fn exponential(initial_rate: f32, decay_rate: f32) -> LearningDecay {
        LearningDecay::new(LearningDecayType::Exponential {
            initial_rate,
            decay_rate,
        })
    }

    pub fn linear_schedule(initial_rate: f32, end_rate: f32, decay_time: usize) -> LearningDecay {
        LearningDecay::new(LearningDecayType::LinearSchedule {
            initial_rate,
            end_rate,
            decay_time,
        })
    }

    pub fn constant(initial_rate: f32) -> LearningDecay {
        LearningDecay::new(LearningDecayType::None { rate: initial_rate })
    }

    pub fn rms_prop(global_rate: f32, decay_rate: f32) -> LearningDecay {
        LearningDecay::new(LearningDecayType::RMSProp {
            global_rate,
            decay_rate,
        })
    }

    fn new(decay_type: LearningDecayType) -> LearningDecay {
        LearningDecay {
            learning_rate: decay_type.get_initial_rate_f32(),
            time_step: 0,
            decay_type: decay_type,
        }
    }

    pub fn get_learning_rate_f32(&self) -> &f32 {
        &self.learning_rate
    }

    pub fn get_learning_rate(&self) -> DataContainer {
        DataContainer::Parameter(Data::ScalarF32(self.learning_rate))
    }

    pub fn next(&mut self) {
        self.learning_rate = self.decay_type.get_next(self.learning_rate, self.time_step);
        self.time_step += 1;
    }
}

#[derive(Serialize, Deserialize)]
pub struct LearningRateParams {
    adaptive_rate: Vec<f32>,
}

impl LearningRateParams {
    pub fn new(adaptive_rate: Vec<f32>) -> LearningRateParams {
        LearningRateParams { adaptive_rate }
    }

    pub fn null() -> LearningRateParams {
        LearningRateParams {
            adaptive_rate: Vec::new(),
        }
    }

    pub fn get_adaptive_learning_rate(&mut self) -> Vec<f32> {
        take(&mut self.adaptive_rate)
    }
}
