use crate::dialect::Item;
use crate::language::{assign, CubeContext, CubeElem, CubeType};
use crate::{unexpanded, ExpandElement};

/// Enable elegant casting from any to any CubeElem
pub trait Cast: CubeElem {
    fn cast_from<From: CubeElem>(value: From) -> Self;

    fn cast_from_expand<From>(
        context: &mut CubeContext,
        value: From,
    ) -> <Self as CubeType>::ExpandType
    where
        From: Into<ExpandElement>,
    {
        let new_var = context.create_local(Item::new(<Self as CubeElem>::as_elem()));
        assign::expand(context, value.into(), new_var.clone());
        new_var
    }
}

impl<P: CubeElem> Cast for P {
    fn cast_from<From: CubeElem>(_value: From) -> Self {
        unexpanded!()
    }
}
