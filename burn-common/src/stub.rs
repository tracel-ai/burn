use spin::{
    Mutex as MutexImported, MutexGuard, RwLock as RwLockImported, RwLockReadGuard, RwLockWriteGuard,
};

// Mutex wrapper to make spin::Mutex API compatible with std::sync::Mutex to swap
#[derive(Debug)]
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

// Mutex wrapper to make spin::Mutex API compatible with std::sync::Mutex to swap
#[derive(Debug)]
pub struct RwLock<T> {
    inner: RwLockImported<T>,
}

impl<T> RwLock<T> {
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: RwLockImported::new(value),
        }
    }

    #[inline(always)]
    pub fn read(&self) -> Result<RwLockReadGuard<T>, alloc::string::String> {
        Ok(self.inner.read())
    }

    #[inline(always)]
    pub fn write(&self) -> Result<RwLockWriteGuard<T>, alloc::string::String> {
        Ok(self.inner.write())
    }
}

// ThreadId stub when no std is available
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(core::num::NonZeroU64);
