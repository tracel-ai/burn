use std::marker::PhantomData;

use crate::{frontend::indexation::Index, unexpanded};

use super::{Array, CubeType, ExpandElementTyped, Init};

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
    pub fn from_array<S: Index>(array: Array<C>, offset: S) -> Self {
        unexpanded!()
    }
}

impl<C: CubeType> ExpandElementTyped<Slice<C>> {
    pub fn from_array_expand<S: Index>(array: ExpandElementTyped<Array<C>>, offset: S) -> Self {
        unexpanded!()
    }
}
