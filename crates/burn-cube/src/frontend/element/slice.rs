use std::marker::PhantomData;

use super::{Array, CubeType, ExpandElement, ExpandElementTyped, Init, Tensor};
use crate::{
    frontend::indexation::Index,
    ir::{self, Operator},
    prelude::CubeContext,
    unexpanded,
};

/// A contiguous list of elements
pub struct Slice<'a, E> {
    _e: PhantomData<E>,
    _l: &'a (),
}

impl<'a, E: CubeType> CubeType for Slice<'a, E> {
    type ExpandType = ExpandElementTyped<Slice<'static, E>>;
}

impl<'a, C: CubeType> Init for ExpandElementTyped<Slice<'a, C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<'a, C: CubeType> Slice<'a, C> {
    #[allow(unused_variables)]
    pub fn from_array<S1: Index, S2: Index>(array: &'a Array<C>, start: S1, end: S2) -> Self {
        unexpanded!()
    }
    pub fn from_array_expand<S1: Index, S2: Index>(
        context: &mut CubeContext,
        array: ExpandElementTyped<Array<C>>,
        start: S1,
        end: S2, // Todo use it to get the length.
    ) -> ExpandElementTyped<Self> {
        ExpandElementTyped::new(slice_expand(context, array, start, end))
    }
}

/// A contiguous list of elements
pub struct SliceMut<'a, E> {
    _e: PhantomData<E>,
    _l: &'a mut (),
}

impl<'a, E: CubeType> CubeType for SliceMut<'a, E> {
    type ExpandType = ExpandElementTyped<SliceMut<'static, E>>;
}

impl<'a, C: CubeType> Init for ExpandElementTyped<SliceMut<'a, C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<'a, C: CubeType> SliceMut<'a, C> {
    #[allow(unused_variables)]
    pub fn from_array<S1: Index, S2: Index>(array: &'a mut Array<C>, start: S1, end: S2) -> Self {
        unexpanded!()
    }
    pub fn from_array_expand<S1: Index, S2: Index>(
        context: &mut CubeContext,
        array: ExpandElementTyped<Array<C>>,
        start: S1,
        end: S2, // Todo use it to get the length.
    ) -> ExpandElementTyped<Self> {
        ExpandElementTyped::new(slice_expand(context, array, start, end))
    }
}

pub fn slice_expand<I: Into<ExpandElement>, S1: Index, S2: Index>(
    context: &mut CubeContext,
    input: I,
    start: S1,
    end: S2, // Todo use it to get the length.
) -> ExpandElement {
    let input = input.into();
    let out = context.create_slice(input.item());

    context.register(Operator::Slice(ir::SliceOperator {
        input: *input,
        offset: start.value(),
        out: *out,
    }));

    out
}
