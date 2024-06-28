use crate::frontend::{assign, CubeContext, CubePrimitive, CubeType};
use crate::ir::Item;
use crate::{frontend::ExpandElement, unexpanded};

/// Enable elegant casting from any to any CubeElem
pub trait Cast: CubePrimitive {
    fn cast_from<From: CubePrimitive>(value: From) -> Self;

    fn cast_from_expand<From>(
        context: &mut CubeContext,
        value: From,
    ) -> <Self as CubeType>::ExpandType
    where
        From: Into<ExpandElement>,
    {
        let new_var = context.create_local(Item::new(<Self as CubePrimitive>::as_elem()));
        assign::expand(context, value.into(), new_var.clone());
        new_var
    }
}

impl<P: CubePrimitive> Cast for P {
    fn cast_from<From: CubePrimitive>(_value: From) -> Self {
        unexpanded!()
    }
}
