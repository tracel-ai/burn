use crate::dialect::Item;
use crate::language::{assign, CubeContext, CubeType, PrimitiveVariable};

// Enable elegant casting from any to any primitive variable

pub trait Cast: PrimitiveVariable {
    fn cast_from_expand(
        context: &mut CubeContext,
        value: <Self as CubeType>::ExpandType,
    ) -> <Self as CubeType>::ExpandType {
        let new_var = context.create_local(Item::scalar(<Self as PrimitiveVariable>::into_elem()));
        assign::expand(context, value, new_var.clone());
        new_var
    }
    fn cast_from<From: PrimitiveVariable>(value: From) -> Self;
}

impl<P: PrimitiveVariable> Cast for P {
    fn cast_from<From: PrimitiveVariable>(value: From) -> Self {
        P::from_f64(value.to_f64())
    }
}
