use crate::frontend::{CubePrimitive, CubeType, ExpandElement};
use crate::ir::Elem;

use super::Vectorized;

// To be consistent with other primitive type.
/// Boolean type.
pub type Bool = bool;

impl CubeType for bool {
    type ExpandType = ExpandElement;
}

impl CubePrimitive for bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}

impl Vectorized for bool {
    fn vectorization_factor(&self) -> crate::prelude::UInt {
        todo!()
    }

    fn vectorize(self, _factor: crate::prelude::UInt) -> Self {
        todo!()
    }
}
