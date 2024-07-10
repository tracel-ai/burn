use std::marker::PhantomData;

use super::{Array, CubeType, ExpandElementTyped, Init};
use crate::{
    frontend::indexation::Index,
    ir::{self, Operator},
    prelude::CubeContext,
    unexpanded,
};

/// A contiguous list of elements
pub struct Slice<E> {
    _e: PhantomData<E>,
}

impl<E: CubeType> CubeType for Slice<E> {
    type ExpandType = ExpandElementTyped<Slice<E>>;
}

impl<C: CubeType> Init for ExpandElementTyped<Slice<C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<C: CubeType> Slice<C> {
    #[allow(unused_variables)]
    pub fn from_array<S1: Index, S2: Index>(array: &Array<C>, start: S1, end: S2) -> Self {
        unexpanded!()
    }
    pub fn from_array_expand<S1: Index, S2: Index>(
        context: &mut CubeContext,
        array: ExpandElementTyped<Array<C>>,
        start: S1,
        end: S2, // Todo use it to get the length.
    ) -> ExpandElementTyped<Self> {
        ExpandElementTyped::<Self>::from_array_expand(context, array, start, end)
    }
}

impl<C: CubeType> ExpandElementTyped<Slice<C>> {
    pub fn from_array_expand<S1: Index, S2: Index>(
        context: &mut CubeContext,
        array: ExpandElementTyped<Array<C>>,
        start: S1,
        end: S2, // Todo use it to get the length.
    ) -> Self {
        let input = *array.expand;
        let out = context.create_slice(input.item());
        context.register(Operator::Slice(ir::SliceOperator {
            input,
            offset: start.value(),
            out: *out,
        }));

        ExpandElementTyped::new(out)
    }
}
