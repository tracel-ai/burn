use crate::{ExpandElement, RuntimeType};

#[derive(new, Clone, Copy)]
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

impl RuntimeType for UInt {
    type ExpandType = ExpandElement;
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value, 1)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32, 1)
    }
}
