use crate::frontend::{CubeElem, CubeType, ExpandElement};
use crate::ir::Elem;

impl CubeType for bool {
    type ExpandType = ExpandElement;
}

impl CubeElem for bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}
