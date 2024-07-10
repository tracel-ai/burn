use std::marker::PhantomData;

use super::{
    Array, CubePrimitive, CubeType, ExpandElement, ExpandElementTyped, Init, SharedMemory, Tensor,
    UInt,
};
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

impl<'a, E> Slice<'a, E> {
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
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

/// A contiguous list of elements
pub struct SliceMut<'a, E> {
    _e: PhantomData<E>,
    _l: &'a mut (),
}

impl<'a, E> SliceMut<'a, E> {
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
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

pub trait SliceOperator<E>: CubeType<ExpandType = Self::Expand> {
    type Expand: SliceOperatorExpand<E>;

    #[allow(unused_variables)]
    fn slice<'a, Start: Index, End: Index>(&'a self, start: Start, end: End) -> Slice<'a, E> {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn slice_mut<'a, Start: Index, End: Index>(
        &'a mut self,
        start: Start,
        end: End,
    ) -> SliceMut<'a, E> {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn slice_mut_unsafe<Start: Index, End: Index>(
        &self,
        start: Start,
        end: End,
    ) -> SliceMut<'static, E> {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn as_slice<'a>(&'a self) -> Slice<'a, E> {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn as_slice_mut<'a>(&'a mut self) -> SliceMut<'a, E> {
        unexpanded!()
    }

    fn slice_expand<Start: Index, End: Index>(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.slice_expand(context, start, end)
    }

    fn slice_mut_expand<Start: Index, End: Index>(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.slice_mut_expand(context, start, end)
    }

    fn as_slice_expand<'a>(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.as_slice_expand(context)
    }

    fn as_slice_unsafe_expand<'a>(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.as_slice_unsafe_expand(context)
    }

    fn as_slice_mut_expand<'a>(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.as_slice_mut_expand(context)
    }
}

pub trait SliceOperatorExpand<E>: Into<ExpandElement> + Clone {
    fn slice_expand<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<Slice<'static, E>>;

    fn slice_mut_expand<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        self.slice_expand(context, start, end)
    }

    fn as_slice_expand(&self, _context: &mut CubeContext) -> ExpandElementTyped<Slice<'static, E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }

    fn as_slice_unsafe_expand(
        &self,
        context: &mut CubeContext,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        self.as_slice_expand(context)
    }

    fn as_slice_mut_expand(
        &self,
        context: &mut CubeContext,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        self.as_slice_expand(context)
    }
}

macro_rules! slice_op {
    ($type:ident) => {
        impl<E: CubePrimitive> SliceOperator<E> for $type<E> {
            type Expand = ExpandElementTyped<$type<E>>;
        }

        impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<E>> {
            fn slice_expand<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElementTyped<Slice<'static, E>> {
                ExpandElementTyped::new(slice_expand(context, self.clone(), start, end))
            }
        }
    };
}

slice_op!(Array);
slice_op!(Tensor);
slice_op!(SharedMemory);

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
        start: start.value(),
        end: end.value(),
        out: *out,
    }));

    out
}
