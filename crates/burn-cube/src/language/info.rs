use crate::{
    dialect::{Elem, Item, Metadata},
    unexpanded, Array, CubeContext, CubeElem, CubeType, ExpandElement, UInt,
};

/// Obtain the stride of input at dimension dim
pub fn stride<T: CubeElem>(_input: Array<T>, _dim: u32) -> UInt {
    unexpanded!()
}

/// Obtain the stride of input at dimension dim
pub fn stride_expand<T: CubeElem>(
    context: &mut CubeContext,
    input: <Array<T> as CubeType>::ExpandType,
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
pub fn shape<T: CubeElem>(_input: Array<T>, _dim: u32) -> UInt {
    unexpanded!()
}

/// Obtain the shape of input at dimension dim
pub fn shape_expand<T: CubeElem>(
    context: &mut CubeContext,
    input: <Array<T> as CubeType>::ExpandType,
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
