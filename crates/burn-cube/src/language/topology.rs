use crate::UInt;

/// In this file we use a trick where the constant has the same name as the module containing
/// the expand function, so that a user implicitly imports the expand function when importing the constant.

/// The index of the working unit in the whole cube kernel, without regards to blocks.
pub const ABSOLUTE_INDEX: UInt = UInt::new(0u32);

#[allow(non_snake_case)]
pub mod ABSOLUTE_INDEX {
    use crate::{CubeContext, ExpandElement};

    /// Expanded version of ABSOLUTE_INDEX
    pub fn expand(_context: &mut CubeContext) -> ExpandElement {
        ExpandElement::Plain(crate::dialect::Variable::Id)
    }
}
