use crate::dialect::Elem;
use crate::language::{CubeType, ExpandElement, CubeElem};

impl CubeType for bool {
    type ExpandType = ExpandElement;
}

impl CubeElem for bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}
