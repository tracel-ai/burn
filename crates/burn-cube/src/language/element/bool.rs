use crate::dialect::{Elem, Vectorization};

use crate::language::{CubeType, ExpandElement, PrimitiveVariable};

// #[derive(Clone, Copy)]
// /// Boolean type for kernels
// pub struct Bool {
//     pub val: <Self as PrimitiveVariable>::Primitive,
//     pub vectorization: u8,
// }

// impl CubeType for Bool {
//     type ExpandType = ExpandElement;
// }

// impl Bool {
//     /// Make a boolean literal
//     pub fn new(val: <Self as PrimitiveVariable>::Primitive) -> Self {
//         Self {
//             val,
//             vectorization: 1,
//         }
//     }

//     /// Expand version of lit
//     pub fn new_expand(
//         _context: &mut CubeContext,
//         val: <Self a PrimitiveVariable>::Primitive,
//     ) -> <Self as CubeType>::ExpandType {
//         val.into()
//     }
// }

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
