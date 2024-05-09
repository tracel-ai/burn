use crate::{ExpandElement, RuntimeType};

#[derive(new, Clone)]
pub struct Array<E> {
    pub vals: Vec<E>,
}

impl<R: RuntimeType> RuntimeType for Array<R> {
    type ExpandType = ExpandElement;
}
