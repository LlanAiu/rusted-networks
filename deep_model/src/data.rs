// builtin

// external
use ndarray::{Array1, Array2, ArrayView2, Axis};

// internal

#[derive(Debug, Clone)]
pub enum Data {
    ScalarF32(f32),
    VectorF32(Array1<f32>),
    MatrixF32(Array2<f32>),
    None,
}

impl Data {
    fn warn_operation(this: &Data, other: &Data, operation: &str) {
        let this_type = this.variant_name();
        let other_type = other.variant_name();
        println!(
            "Data::None returned on unsupported data type pair for operation [{operation}]: {this_type} and {other_type}!"
        );
    }
}

impl Data {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Data::ScalarF32(_) => "ScalarF32",
            Data::VectorF32(_) => "VectorF32",
            Data::MatrixF32(_) => "MatrixF32",
            Data::None => "None",
        }
    }

    pub fn sum(&self, other: &Data) -> Data {
        match self {
            Data::ScalarF32(this) => {
                if let Data::ScalarF32(scalar) = other {
                    return Data::ScalarF32(scalar + this);
                }
                Data::warn_operation(self, other, "SUM");
                Data::None
            }
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this + vec);
                    }
                }
                Data::warn_operation(self, other, "SUM");
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this + matrix);
                    }
                }
                Data::warn_operation(self, other, "SUM");
                Data::None
            }
            Data::None => other.clone(),
        }
    }

    pub fn minus(&self, other: &Data) -> Data {
        match self {
            Data::ScalarF32(this) => {
                if let Data::ScalarF32(scalar) = other {
                    return Data::ScalarF32(this - scalar);
                }
                Data::warn_operation(self, other, "MINUS");
                Data::None
            }
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this - vec);
                    }
                }
                Data::warn_operation(self, other, "MINUS");
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this - matrix);
                    }
                }
                Data::warn_operation(self, other, "MINUS");
                Data::None
            }
            Data::None => other.scale_f32(-1.0),
        }
    }

    pub fn times(&self, other: &Data) -> Data {
        match self {
            Data::ScalarF32(this) => {
                if let Data::ScalarF32(scalar) = other {
                    return Data::ScalarF32(this * scalar);
                }
                Data::warn_operation(self, other, "TIMES");
                Data::None
            }
            Data::VectorF32(this) => {
                if let Data::VectorF32(vec) = other {
                    if vec.dim() == this.dim() {
                        return Data::VectorF32(this * vec);
                    }
                }
                Data::warn_operation(self, other, "TIMES");
                Data::None
            }
            Data::MatrixF32(this) => {
                if let Data::MatrixF32(matrix) = other {
                    if matrix.dim() == this.dim() {
                        return Data::MatrixF32(this * matrix);
                    }
                }
                Data::warn_operation(self, other, "TIMES");
                Data::None
            }
            Data::None => {
                Data::warn_operation(self, other, "TIMES");
                Data::None
            }
        }
    }

    pub fn dot(&self, other: &Data) -> Data {
        let other_matrix = match other {
            Data::VectorF32(vec) => vec.view().insert_axis(Axis(1)),
            Data::MatrixF32(matrix) => matrix.view(),
            _ => {
                Data::warn_operation(self, other, "DOT");
                return Data::None;
            }
        };

        let self_matrix: ArrayView2<f32> = match self {
            Data::VectorF32(vec) => vec.view().insert_axis(Axis(1)),
            Data::MatrixF32(matrix) => matrix.view(),
            _ => {
                Data::warn_operation(self, other, "DOT");
                return Data::None;
            }
        };

        let res = self_matrix.dot(&other_matrix);

        if res.dim().1 == 1 {
            return Data::VectorF32(res.remove_axis(Axis(1)));
        }

        Data::MatrixF32(res)
    }

    pub fn scale_f32(&self, scalar: f32) -> Data {
        match self {
            Data::ScalarF32(this) => Data::ScalarF32(this * scalar),
            Data::VectorF32(this) => Data::VectorF32(this * scalar),
            Data::MatrixF32(this) => Data::MatrixF32(this * scalar),
            Data::None => Data::None,
        }
    }

    pub fn scale(&self, other: &Data) -> Data {
        if let Data::ScalarF32(scalar) = other {
            return match self {
                Data::ScalarF32(this) => Data::ScalarF32(this * scalar),
                Data::VectorF32(this) => Data::VectorF32(this * *scalar),
                Data::MatrixF32(this) => Data::MatrixF32(this * *scalar),
                Data::None => Data::None,
            };
        }
        Data::warn_operation(self, other, "SCALE");
        Data::None
    }
}

impl Default for Data {
    fn default() -> Self {
        Data::None
    }
}
