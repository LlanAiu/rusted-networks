// builtin

// external
use ndarray::{Array1, Array2};

// internal

#[derive(Debug, Clone)]
pub enum Data {
    VectorF32(Array1<f32>),
    MatrixF32(Array2<f32>),
    None,
}

impl Data {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Data::VectorF32(_) => "VectorF32",
            Data::MatrixF32(_) => "MatrixF32",
            Data::None => "None",
        }
    }
}

impl Data {
    pub fn sum(&self, other: &Data) -> Data {
        match self {
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this + vec);
                    }
                }
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this + matrix);
                    }
                }
                Data::None
            }
            Data::None => other.clone(),
        }
    }

    pub fn minus(&self, other: &Data) -> Data {
        match self {
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this - vec);
                    }
                }
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this - matrix);
                    }
                }
                Data::None
            }
            Data::None => other.clone().scale(-1.0),
        }
    }

    pub fn times(&self, other: &Data) -> Data {
        match self {
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this * vec);
                    }
                }
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this * matrix);
                    }
                }
                Data::None
            }
            Data::None => Data::None,
        }
    }

    pub fn scale(&self, scalar: f32) -> Data {
        match self {
            Data::VectorF32(this) => Data::VectorF32(this * scalar),
            Data::MatrixF32(this) => Data::MatrixF32(this * scalar),
            Data::None => Data::None,
        }
    }
}

impl Default for Data {
    fn default() -> Self {
        Data::None
    }
}
