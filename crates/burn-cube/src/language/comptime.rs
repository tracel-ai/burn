use crate::{unexpanded, CubeContext, CubeType};

#[derive(Clone, Copy)]
/// Encapsulates a value to signify it must be used at compilation time rather than in the kernel
pub struct Comptime<T> {
    inner: T,
}

impl<T> Comptime<T> {
    pub fn new(_inner: T) -> Self {
        unexpanded!()
    }

    pub fn get(_comptime: Self) -> T {
        unexpanded!()
    }
}

impl<T: CubeType + Into<T::ExpandType>> Comptime<Option<T>> {
    pub fn is_some(comptime: Self) -> Comptime<bool> {
        Comptime::new(comptime.inner.is_some())
    }

    pub fn unwrap_or_else<F>(_comptime: Self, mut _alt: F) -> T
    where
        F: FnMut() -> T,
    {
        unexpanded!()
    }

    pub fn unwrap_or_else_expand<F>(
        context: &mut CubeContext,
        t: Option<T>,
        mut alt: F,
    ) -> <T as CubeType>::ExpandType
    where
        F: FnMut(&mut CubeContext) -> T::ExpandType,
    {
        match t {
            Some(t) => t.into(),
            None => alt(context),
        }
    }
}

impl<T: Clone> CubeType for Comptime<T> {
    type ExpandType = T;
}
