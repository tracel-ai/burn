//! A `Sync`-compatible single-initialization cell for both `std` and `no_std`.
//!
//! Wraps `std::sync::OnceLock` (with `std`) or `spin::Once` (without `std`) behind
//! a unified API. This makes `Param<T>` `Sync` so models can be shared across threads
//! for parallel inference.
//!
//! We define our own wrapper instead of reusing `burn_std::stub::SyncOnceCell` because
//! that version requires `T: Debug` on all methods and lacks `get()`.

#[cfg(feature = "std")]
use std::sync::OnceLock as Inner;

#[cfg(not(feature = "std"))]
use spin::Once as Inner;

pub(crate) struct SyncOnceCell<T>(Inner<T>);

impl<T: core::fmt::Debug> core::fmt::Debug for SyncOnceCell<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SyncOnceCell").field(&self.get()).finish()
    }
}

impl<T> SyncOnceCell<T> {
    /// Create a new empty cell.
    pub fn new() -> Self {
        Self(Inner::new())
    }

    /// Create a cell pre-populated with `value`.
    pub fn initialized(value: T) -> Self {
        #[cfg(feature = "std")]
        {
            let cell = Inner::new();
            cell.set(value).ok().expect("cell was just created");
            Self(cell)
        }
        #[cfg(not(feature = "std"))]
        {
            Self(Inner::initialized(value))
        }
    }

    /// Returns `Some(&T)` if initialized, `None` otherwise.
    pub fn get(&self) -> Option<&T> {
        self.0.get()
    }

    /// Returns `&T`, initializing with `f` on first call.
    pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
        #[cfg(feature = "std")]
        {
            self.0.get_or_init(f)
        }
        #[cfg(not(feature = "std"))]
        {
            self.0.call_once(f)
        }
    }
}
