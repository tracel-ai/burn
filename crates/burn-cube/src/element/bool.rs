use crate::{ExpandElement, RuntimeType};

#[derive(new, Clone, Copy)]
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

impl RuntimeType for Bool {
    type ExpandType = ExpandElement;
}
