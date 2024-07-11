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

/// A read-only contiguous list of elements
pub struct Slice<'a, E> {
    _e: PhantomData<E>,
    _l: &'a (),
}

/// A read-write contiguous list of elements.
pub struct SliceMut<'a, E> {
    _e: PhantomData<E>,
    _l: &'a mut (),
}

impl<'a, E> Slice<'a, E> {
    /// Get the length of the slice.
    pub fn len(&self) -> UInt {
        unexpanded!()
    }
}

impl<'a, E> SliceMut<'a, E> {
    /// Get the length of the slice.
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

    /// Return a read-only view of all elements comprise between the start and end index.
    #[allow(unused_variables)]
    fn slice<Start: Index, End: Index>(&self, start: Start, end: End) -> &'_ Slice<'_, E> {
        unexpanded!()
    }
    /// Expand function of [SliceOperator::slice].
    fn slice_expand<Start: Index, End: Index>(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.slice_expand(context, start, end)
    }

    /// Return a read-write view of all elements comprise between the start and end index.
    #[allow(unused_variables)]
    fn slice_mut<Start: Index, End: Index>(
        &mut self,
        start: Start,
        end: End,
    ) -> &'_ mut SliceMut<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::slice_mut].
    fn slice_mut_expand<Start: Index, End: Index>(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.slice_mut_expand(context, start, end)
    }

    /// Return a read-write view of all elements comprise between the start and end index.
    ///
    /// # Warning
    ///
    /// Ignore the multiple borrow rule.
    #[allow(unused_variables)]
    fn slice_mut_unsafe<Start: Index, End: Index>(
        &self,
        start: Start,
        end: End,
    ) -> SliceMut<'static, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::slice_mut_unsafe].
    fn slice_mut_unsafe_expand<Start: Index, End: Index>(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.slice_mut_unsafe_expand(context, start, end)
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn as_slice(&self) -> &'_ Slice<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::as_slice].
    fn as_slice_expand(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        expand.as_slice_expand(context)
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables)]
    fn as_slice_mut(&mut self) -> &'_ mut SliceMut<'_, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::as_slice_mut].
    fn as_slice_mut_expand(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.as_slice_mut_expand(context)
    }

    /// Reinterprete the current type as a read-write slice.
    ///
    /// # Warning
    ///
    /// Ignore the multiple borrow rule.
    #[allow(unused_variables)]
    fn as_slice_mut_unsafe(&self) -> SliceMut<'static, E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::as_slice_mut_unsafe].
    fn as_slice_mut_unsafe_expand(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        expand.as_slice_mut_unsafe_expand(context)
    }
}

pub trait SliceOperatorExpand<E>: Into<ExpandElement> + Clone {
    fn slice_base<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElement;

    fn slice_expand<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<Slice<'static, E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn slice_mut_expand<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn slice_mut_unsafe_expand<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn as_slice_expand(&self, _context: &mut CubeContext) -> ExpandElementTyped<Slice<'static, E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }

    fn as_slice_mut_unsafe_expand(
        &self,
        context: &mut CubeContext,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        self.as_slice_mut_expand(context)
    }

    fn as_slice_mut_expand(
        &self,
        _context: &mut CubeContext,
    ) -> ExpandElementTyped<SliceMut<'static, E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }
}

macro_rules! slice_op {
    ($type:ident) => {
        impl<E: CubePrimitive> SliceOperator<E> for $type<E> {
            type Expand = ExpandElementTyped<$type<E>>;
        }

        impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<E>> {
            fn slice_base<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElement {
                slice_expand(context, self.clone(), start, end)
            }
        }
    };
    (slice $type:ident) => {
        impl<'a, E: CubePrimitive> SliceOperator<E> for $type<'a, E> {
            type Expand = ExpandElementTyped<$type<'static, E>>;
        }

        impl<'a, E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<'a, E>> {
            fn slice_base<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElement {
                slice_expand(context, self.clone(), start, end)
            }
        }
    };
}

slice_op!(Array);
slice_op!(Tensor);
slice_op!(SharedMemory);
slice_op!(slice Slice);
slice_op!(slice SliceMut);

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
