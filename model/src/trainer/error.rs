// builtin

// external

// internal

#[derive(Clone, Debug)]
pub enum PredictionError {
    Loss { loss: f32 },
    Misclassification { incorrect: usize, total: usize },
    Empty,
    None,
}

impl PredictionError {
    fn warn() {
        println!("Got mismatched PredictionError types, returning PredictionError::None");
    }

    pub fn empty() -> PredictionError {
        PredictionError::Empty
    }

    pub fn plus(&self, other: &PredictionError) -> PredictionError {
        match (self, other) {
            (PredictionError::Loss { loss }, PredictionError::Loss { loss: second_loss }) => {
                PredictionError::Loss {
                    loss: loss + second_loss,
                }
            }
            (
                PredictionError::Misclassification {
                    incorrect: l_incorrect,
                    total: l_total,
                },
                PredictionError::Misclassification {
                    incorrect: r_incorrect,
                    total: r_total,
                },
            ) => PredictionError::Misclassification {
                incorrect: l_incorrect + r_incorrect,
                total: l_total + r_total,
            },
            (PredictionError::Empty, _) => other.clone(),
            (_, PredictionError::Empty) => self.clone(),
            _ => {
                PredictionError::warn();
                PredictionError::None
            }
        }
    }
}

impl PartialEq for PredictionError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Loss { loss: l_loss }, Self::Loss { loss: r_loss }) => l_loss == r_loss,
            (
                Self::Misclassification {
                    incorrect: l_incorrect,
                    total: l_total,
                },
                Self::Misclassification {
                    incorrect: r_incorrect,
                    total: r_total,
                },
            ) => {
                let l_error_rate = if *l_total > 0 {
                    (*l_incorrect as f32) / (*l_total as f32)
                } else {
                    0.0
                };

                let r_error_rate = if *r_total > 0 {
                    (*r_incorrect as f32) / (*r_total as f32)
                } else {
                    0.0
                };

                return l_error_rate == r_error_rate;
            }
            _ => false,
        }
    }
}

impl PartialOrd for PredictionError {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (PredictionError::Loss { loss: l_loss }, PredictionError::Loss { loss: r_loss }) => {
                f32::partial_cmp(l_loss, r_loss)
            }
            (
                PredictionError::Misclassification {
                    incorrect: l_incorrect,
                    total: l_total,
                },
                PredictionError::Misclassification {
                    incorrect: r_incorrect,
                    total: r_total,
                },
            ) => {
                let l_error_rate = if *l_total > 0 {
                    (*l_incorrect as f32) / (*l_total as f32)
                } else {
                    0.0
                };

                let r_error_rate = if *r_total > 0 {
                    (*r_incorrect as f32) / (*r_total as f32)
                } else {
                    0.0
                };

                f32::partial_cmp(&l_error_rate, &r_error_rate)
            }

            _ => None,
        }
    }
}
