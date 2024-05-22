use crate::{CubeContext, CubeType};

#[derive(Clone, Copy)]
/// Encapsulates a value to signify it must be used at compilation time rather than in the kernel
pub struct Comptime<T> {
    inner: T,
}

impl<T> Comptime<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    pub fn get(comptime: Self) -> T {
        comptime.inner
    }
}

impl<T: CubeType + Into<T::ExpandType>> Comptime<Option<T>> {
    pub fn is_some(comptime: Self) -> Comptime<bool> {
        Comptime::new(comptime.inner.is_some())
    }
    pub fn value_or<F>(comptime: Self, mut alt: F) -> T
    where
        F: FnMut() -> T,
    {
        match comptime.inner {
            Some(t) => t,
            None => alt(),
        }
    }

    pub fn value_or_expand<F>(
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
