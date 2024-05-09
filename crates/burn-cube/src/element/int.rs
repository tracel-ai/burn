use crate::{ExpandElement, RuntimeType};

#[derive(new, Clone, Copy)]
pub struct Int {
    pub val: i32,
    pub vectorization: u8,
}

impl RuntimeType for Int {
    type ExpandType = ExpandElement;
}

impl From<i32> for Int {
    fn from(value: i32) -> Self {
        Int::new(value, 1)
    }
}
