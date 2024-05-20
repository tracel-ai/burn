use crate::dialect::{Elem, Vectorization};
use crate::language::{CubeType, ExpandElement, PrimitiveVariable};

impl CubeType for bool {
    type ExpandType = ExpandElement;
}

impl PrimitiveVariable for bool {
    type Primitive = bool;

    fn as_elem() -> Elem {
        Elem::Bool
    }

    fn vectorization(&self) -> Vectorization {
        1
    }

    fn to_f64(&self) -> f64 {
        match self {
            true => 1.,
            false => 0.,
        }
    }

    fn from_f64(val: f64) -> Self {
        val > 0.
    }

    fn from_i64(val: i64) -> Self {
        val > 0
    }

    fn from_i64_vec(_vec: &[i64]) -> Self {
        panic!()
    }
}
