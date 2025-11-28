// builtin

// external

// internal

pub enum FlattenedData {
    Batch(Vec<Vec<f32>>),
    Singular(Vec<f32>),
    None,
}
