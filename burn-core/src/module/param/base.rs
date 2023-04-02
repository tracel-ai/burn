use alloc::format;
use serde::{Deserialize, Serialize};

use super::ParamId;

/// Define a trainable parameter.
#[derive(new, Debug, Clone, Serialize, Deserialize)]
pub struct Param<T> {
    pub(crate) id: ParamId,
    pub(crate) value: T,
}

impl<T> core::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T: Clone> Param<T> {
    pub fn val(&self) -> T {
        self.value.clone()
    }
}

impl<T> core::ops::Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
