use spin::{Mutex as MutexImported, MutexGuard};

// Mutex wrapper to make spin::Mutex API compatible with std::sync::Mutex to swap
pub struct Mutex<T> {
    inner: MutexImported<T>,
}

impl<T> Mutex<T> {
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: MutexImported::new(value),
        }
    }
    #[inline(always)]
    pub fn lock(&self) -> Result<MutexGuard<T>, alloc::string::String> {
        Ok(self.inner.lock())
    }
}
