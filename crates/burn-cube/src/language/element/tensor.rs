use std::marker::PhantomData;

use crate::{
    dialect::{Elem, Item, Metadata},
    language::{CubeType, ExpandElement},
    unexpanded, CubeContext, UInt,
};

#[derive(new, Clone, Copy)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for Tensor<T> {
    type ExpandType = ExpandElement;
}

impl<T: CubeType> Tensor<T> {
    /// Obtain the stride of input at dimension dim
    pub fn stride(_input: Tensor<T>, _dim: u32) -> UInt {
        unexpanded!()
    }

    /// Obtain the stride of input at dimension dim
    pub fn stride_expand(
        context: &mut CubeContext,
        input: <Tensor<T> as CubeType>::ExpandType,
        dim: u32,
    ) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Stride {
            dim: dim.into(),
            var: input.into(),
            out: out.clone().into(),
        });
        out
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape(_input: Tensor<T>, _dim: u32) -> UInt {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape_expand(
        context: &mut CubeContext,
        input: <Tensor<T> as CubeType>::ExpandType,
        dim: u32,
    ) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Shape {
            dim: dim.into(),
            var: input.into(),
            out: out.clone().into(),
        });
        out
    }

    pub fn len(_input: Self) -> UInt {
        unexpanded!()
    }

    pub fn len_expand(
        context: &mut CubeContext,
        input: <Tensor<T> as CubeType>::ExpandType,
    ) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::ArrayLength {
            var: input.into(),
            out: out.clone().into(),
        });
        out
    }
}
