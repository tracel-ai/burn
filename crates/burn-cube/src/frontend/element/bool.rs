use crate::frontend::{CubeElem, CubeType, ExpandElement};
use crate::ir::Elem;

use super::Vectorized;

impl CubeType for bool {
    type ExpandType = ExpandElement;
}

impl CubeElem for bool {
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
