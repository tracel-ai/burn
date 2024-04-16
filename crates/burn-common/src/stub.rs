#[cfg(not(feature = "std"))]
use spin::{
    Mutex as MutexImported, MutexGuard, Once as OnceImported, RwLock as RwLockImported,
    RwLockReadGuard, RwLockWriteGuard,
};
#[cfg(feature = "std")]
use std::sync::{
    Mutex as MutexImported, MutexGuard, OnceLock as OnceImported, RwLock as RwLockImported,
    RwLockReadGuard, RwLockWriteGuard,
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
        #[cfg(not(feature = "std"))]
        {
            Ok(self.inner.lock())
        }

        #[cfg(feature = "std")]
        {
            self.inner.lock().map_err(|err| err.to_string())
        }
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
        #[cfg(not(feature = "std"))]
        {
            Ok(self.inner.read())
        }
        #[cfg(feature = "std")]
        {
            self.inner.read().map_err(|err| err.to_string())
        }
    }

    /// Locks this rwlock with exclusive write access, blocking the current thread
    /// until it can be acquired.
    #[inline(always)]
    pub fn write(&self) -> Result<RwLockWriteGuard<T>, alloc::string::String> {
        #[cfg(not(feature = "std"))]
        {
            Ok(self.inner.write())
        }

        #[cfg(feature = "std")]
        {
            self.inner.write().map_err(|err| err.to_string())
        }
    }
}

/// A unique identifier for a running thread.
///
/// This module is a stub when no std is available to swap with std::thread::ThreadId.
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(core::num::NonZeroU64);

/// A cell that provides lazy one-time initialization that implements [Sync] and [Send].
///
/// This module is a stub when no std is available to swap with [std::sync::OnceLock].
pub struct SyncOnceCell<T>(OnceImported<T>);

impl<T: core::fmt::Debug> Default for SyncOnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: core::fmt::Debug> SyncOnceCell<T> {
    /// Create a new once.
    #[inline(always)]
    pub fn new() -> Self {
        Self(OnceImported::new())
    }

    /// Initialize the cell with a value.
    #[inline(always)]
    pub fn initialized(value: T) -> Self {
        #[cfg(not(feature = "std"))]
        {
            let cell = OnceImported::initialized(value);
            Self(cell)
        }

        #[cfg(feature = "std")]
        {
            let cell = OnceImported::new();
            cell.set(value).unwrap();

            Self(cell)
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell
    /// was empty.
    #[inline(always)]
    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        #[cfg(not(feature = "std"))]
        {
            self.0.call_once(f)
        }

        #[cfg(feature = "std")]
        {
            self.0.get_or_init(f)
        }
    }
}
