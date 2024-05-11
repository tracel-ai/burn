use crate::{CubeType, ExpandElement};

#[derive(new, Clone)]
pub struct Array<E> {
    pub vals: Vec<E>,
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElement;
}
