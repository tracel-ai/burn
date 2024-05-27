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
    pub fn stride(self, _dim: u32) -> UInt {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape(self, _dim: u32) -> UInt {
        unexpanded!()
    }

    /// Obtain the array length of input
    pub fn len(self) -> UInt {
        unexpanded!()
    }
}

impl ExpandElement {
    // Expanded version of Tensor::stride
    pub fn stride_expand(self, context: &mut CubeContext, dim: u32) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Stride {
            dim: dim.into(),
            var: self.into(),
            out: out.clone().into(),
        });
        out
    }

    // Expanded version of Tensor::shape
    pub fn shape_expand(self, context: &mut CubeContext, dim: u32) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Shape {
            dim: dim.into(),
            var: self.into(),
            out: out.clone().into(),
        });
        out
    }

    // Expanded version of Tensor::len
    pub fn len_expand(self, context: &mut CubeContext) -> ExpandElement {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::ArrayLength {
            var: self.into(),
            out: out.clone().into(),
        });
        out
    }
}
