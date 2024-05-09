use crate::{Bool, ExpandElement, Float, Int, RuntimeType, UInt};

#[derive(new, Clone)]
pub struct Array<E> {
    pub vals: Vec<E>,
}

impl<F: Float> RuntimeType for Array<F> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<Int> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<UInt> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<Bool> {
    type ExpandType = ExpandElement;
}
