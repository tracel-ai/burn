use crate::{ExpandElement, CubeType};

#[derive(Clone, Copy)]
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

impl UInt {
    pub fn new(val: u32) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }
}

impl CubeType for UInt {
    type ExpandType = ExpandElement;
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32)
    }
}
