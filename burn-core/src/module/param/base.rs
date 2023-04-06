use super::ParamId;
use alloc::format;
use core::marker::PhantomData;

/// Type flag for tracked parameters.
#[derive(new, Debug, Clone)]
pub struct Tracked;

/// Type flag for untracked parameters.
#[derive(new, Debug, Clone)]
pub struct Untracked;

/// Define a parameter.
#[derive(new, Debug, Clone)]
pub struct Param<T, K = Tracked> {
    pub(crate) id: ParamId,
    pub(crate) value: T,
    phantom: PhantomData<K>,
}

impl<T> core::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T: Clone, K> Param<T, K> {
    pub fn val(&self) -> T {
        self.value.clone()
    }
}

impl<T, K> core::ops::Deref for Param<T, K> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
