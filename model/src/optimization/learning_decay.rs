// builtin

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
    RMSProp {
        global_rate: f32,
        decay_rate: f32,
    },
    LinearSchedule {
        start_rate: f32,
        end_rate: f32,
        end_time: usize,
        time: usize,
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

    pub fn constant(initial_rate: f32) -> LearningDecayType {
        LearningDecayType::None { rate: initial_rate }
    }

    pub fn rms_prop(global_rate: f32, decay_rate: f32) -> LearningDecayType {
        LearningDecayType::RMSProp {
            global_rate,
            decay_rate,
        }
    }

    pub fn is_adaptive(&self) -> bool {
        match self {
            LearningDecayType::Exponential { .. } => false,
            LearningDecayType::RMSProp { .. } => true,
            LearningDecayType::None { .. } => false,
            LearningDecayType::LinearSchedule { .. } => false,
        }
    }

    pub fn get_initial_timestep(&self) -> usize {
        match self {
            LearningDecayType::LinearSchedule { time, .. } => *time,
            _ => 0,
        }
    }

    pub fn get_initial_rate(&self) -> DataContainer {
        match self {
            LearningDecayType::Exponential { initial_rate, .. } => {
                DataContainer::Parameter(Data::ScalarF32(*initial_rate))
            }
            LearningDecayType::LinearSchedule {
                start_rate,
                end_rate,
                end_time,
                time,
            } => {
                let percent = (*time as f32) / (*end_time as f32);
                let rate = (1.0 - percent) * *start_rate + percent * *end_rate;

                DataContainer::Parameter(Data::ScalarF32(rate))
            }
            LearningDecayType::RMSProp { .. } => DataContainer::zero(),
            LearningDecayType::None { rate } => DataContainer::Parameter(Data::ScalarF32(*rate)),
        }
    }

    pub fn update_timestep(&mut self, time_step: usize) {
        if let LearningDecayType::LinearSchedule {
            start_rate,
            end_rate,
            end_time,
            ..
        } = self
        {
            *self = LearningDecayType::LinearSchedule {
                start_rate: *start_rate,
                end_rate: *end_rate,
                end_time: *end_time,
                time: time_step,
            }
        }
    }

    pub fn update_global(&mut self, learning_rate: &mut DataContainer, time_step: usize) {
        self.update_timestep(time_step);
        match &self {
            LearningDecayType::Exponential { decay_rate, .. } => {
                learning_rate.apply_inplace(|f| *f *= decay_rate);
            }
            LearningDecayType::LinearSchedule {
                start_rate,
                end_rate,
                end_time,
                time,
            } => {
                let percent = (*time as f32) / (*end_time as f32);
                let rate = (1.0 - percent) * *start_rate + percent * *end_rate;

                learning_rate.apply_inplace(|f| *f = rate);
            }
            LearningDecayType::None { .. } => {}
            _ => {
                println!("Cannot compute global update on an adaptive learning rate configuration, try using .update_adaptive() instead");
            }
        }
    }

    pub fn update_adaptive(
        &mut self,
        learning_rate: &mut DataContainer,
        gradient: &DataContainer,
        time_step: usize,
    ) {
        self.update_timestep(time_step);
        match &self {
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

#[derive(Serialize, Deserialize)]
pub struct LearningRateParams {
    adaptive_rate: Vec<f32>,
}

impl LearningRateParams {
    pub fn new(learning_rate: Vec<f32>) -> LearningRateParams {
        LearningRateParams {
            adaptive_rate: learning_rate,
        }
    }

    pub fn null() -> LearningRateParams {
        LearningRateParams {
            adaptive_rate: Vec::new(),
        }
    }

    pub fn get_adaptive_learning_rate(&self) -> Vec<f32> {
        self.adaptive_rate.clone()
    }
}
