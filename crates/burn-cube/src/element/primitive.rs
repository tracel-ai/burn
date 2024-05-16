use std::rc::Rc;

use burn_jit::gpu::{Elem, Item, Variable};

use crate::{assign, CubeContext, CubeType, ExpandElement};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
pub trait PrimitiveVariable: CubeType<ExpandType = ExpandElement> {
    /// Type of the value kept CPU-side.
    /// Does not necessarily match the GPU type.
    type Primitive;

    /// Return the value of the float on CPU
    fn val(&self) -> Self::Primitive;

    /// Return the element type to use on GPU
    fn into_elem() -> Elem;

    /// Expand version of from, of the trait From
    fn from_expand(
        context: &mut CubeContext,
        val: ExpandElement,
    ) -> <Self as CubeType>::ExpandType {
        let new_var = context.create_local(Item::Scalar(<Self as PrimitiveVariable>::into_elem()));
        assign::expand(context, val, new_var.clone());
        new_var
    }
}

macro_rules! impl_into_expand_element {
    ($type:ty) => {
        impl From<$type> for ExpandElement {
            fn from(value: $type) -> Self {
                ExpandElement::new(Rc::new(Variable::from(value)))
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
