use crate::{CubeType, ExpandElement};

#[derive(new, Clone, Copy)]
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

impl CubeType for Bool {
    type ExpandType = ExpandElement;
}
