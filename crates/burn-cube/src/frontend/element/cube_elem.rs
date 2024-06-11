use crate::frontend::UInt;
use crate::frontend::{CubeType, ExpandElement};
use crate::ir::{Elem, Variable};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
pub trait CubeElem:
    CubeType<ExpandType = ExpandElement>
    + core::cmp::Eq
    + core::cmp::PartialEq
    + Send
    + Sync
    + 'static
    + Clone
{
    /// Return the element type to use on GPU
    fn as_elem() -> Elem;
}

macro_rules! impl_into_expand_element {
    ($type:ty) => {
        impl From<$type> for ExpandElement {
            fn from(value: $type) -> Self {
                ExpandElement::Plain(Variable::from(value))
            }
        }
    };
}

impl_into_expand_element!(u32);
impl_into_expand_element!(usize);
impl_into_expand_element!(bool);
impl_into_expand_element!(f32);
impl_into_expand_element!(i32);
impl_into_expand_element!(i64);

/// Useful for Comptime
impl From<UInt> for ExpandElement {
    fn from(value: UInt) -> Self {
        ExpandElement::Plain(crate::ir::Variable::ConstantScalar(
            value.val as f64,
            UInt::as_elem(),
        ))
    }
}
