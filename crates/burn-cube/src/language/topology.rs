use crate::{unexpanded, CubeContext, CubeType, ExpandElement, UInt};

/// The index of the working unit in the whole cube kernel, without regards to blocks.
pub struct AbsoluteIndex {}

impl AbsoluteIndex {
    /// Obtain the absolute index
    pub fn get() -> UInt {
        unexpanded!();
    }

    /// Obtain the absolute index
    pub fn get_expand(_context: &mut CubeContext) -> ExpandElement {
        ExpandElement::Plain(crate::dialect::Variable::Id)
    }
}

impl CubeType for AbsoluteIndex {
    type ExpandType = ExpandElement;
}
