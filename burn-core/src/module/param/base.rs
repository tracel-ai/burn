use super::ParamId;
use alloc::format;

/// Define a parameter.
#[derive(new, Debug, Clone)]
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
    /// Gets the parameter value.
    ///
    /// # Returns
    ///
    /// The parameter value.
    pub fn val(&self) -> T {
        self.value.clone()
    }

    /// Execute the given function on the inner value.
    pub fn map<F: FnOnce(T) -> T>(mut self, func: F) -> Self {
        self.value = func(self.value);
        self
    }
}

impl<T> core::ops::Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
