use burn_jit::gpu::{Elem, Item};

use crate::{assign, CubeContext, CubeType, ExpandElement};

pub trait RuntimeType: CubeType<ExpandType = ExpandElement> {
    type Primitive;
    fn val(&self) -> Self::Primitive;
    fn into_elem() -> Elem;
    fn from_expand(context: &mut CubeContext, val: ExpandElement) -> ExpandElement {
        let new_var = context.create_local(Item::Scalar(<Self as RuntimeType>::into_elem()));
        assign::expand(context, val.into(), new_var.clone());
        new_var
    }
}
