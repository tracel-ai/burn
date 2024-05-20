use crate::{CubeContext, CubeType, ExpandElement, UInt};

pub struct AbsoluteIndex {}

impl AbsoluteIndex {
    pub fn get() -> UInt {
        UInt::new(0u32)
    }
    pub fn get_expand(_context: &mut CubeContext) -> ExpandElement {
        ExpandElement::Plain(crate::dialect::Variable::Id)
    }
}

impl CubeType for AbsoluteIndex {
    type ExpandType = ExpandElement;
}
