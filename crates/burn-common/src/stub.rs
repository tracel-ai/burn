use spin::{
    Mutex as MutexImported, MutexGuard, RwLock as RwLockImported, RwLockReadGuard, RwLockWriteGuard,
};

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can also be statically initialized or created via a [Mutex::new]
///
/// [Mutex] wrapper to make `spin::Mutex` API compatible with `std::sync::Mutex` to swap
#[derive(Debug)]
pub struct Mutex<T> {
    inner: MutexImported<T>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: MutexImported::new(value),
        }
    }

    /// Locks the mutex blocking the current thread until it is able to do so.
    #[inline(always)]
    pub fn lock(&self) -> Result<MutexGuard<T>, alloc::string::String> {
        Ok(self.inner.lock())
    }
}

/// A reader-writer lock which is exclusively locked for writing or shared for reading.
/// This reader-writer lock will block threads waiting for the lock to become available.
/// The lock can also be statically initialized or created via a [RwLock::new]
/// [RwLock] wrapper to make `spin::RwLock` API compatible with `std::sync::RwLock` to swap
#[derive(Debug)]
pub struct RwLock<T> {
    inner: RwLockImported<T>,
}

impl<T> RwLock<T> {
    /// Creates a new reader-writer lock in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: RwLockImported::new(value),
        }
    }

    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    #[inline(always)]
    pub fn read(&self) -> Result<RwLockReadGuard<T>, alloc::string::String> {
        Ok(self.inner.read())
    }

    /// Locks this rwlock with exclusive write access, blocking the current thread
    /// until it can be acquired.
    #[inline(always)]
    pub fn write(&self) -> Result<RwLockWriteGuard<T>, alloc::string::String> {
        Ok(self.inner.write())
    }
}

/// A unique identifier for a running thread.
///
/// This module is a stub when no std is available to swap with std::thread::ThreadId.
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(core::num::NonZeroU64);
