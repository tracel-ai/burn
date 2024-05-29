use crate::UInt;

/// The index of the working unit in the whole cube kernel, without regards to blocks.
pub const ABSOLUTE_INDEX: UInt = UInt::new(0u32);

#[allow(non_snake_case)]
pub mod ABSOLUTE_INDEX {
    use crate::{CubeContext, ExpandElement};
    pub fn expand(_context: &mut CubeContext) -> ExpandElement {
        ExpandElement::Plain(crate::dialect::Variable::Id)
    }
}
