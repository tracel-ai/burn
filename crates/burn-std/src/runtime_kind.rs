//! Runtime kind of the host program.
//!
//! Some backend decisions depend not on the device but on *how the host program itself is
//! being driven* — whether `main` runs on an asynchronous runtime, a synchronous
//! thread-based runtime, or a restricted no-std environment. This module stores that kind
//! in a process global so backends can read it and adapt their behavior.
//!
//! The canonical example is tensor readback (`into_data`): deferring the device→host copy
//! lazily is fine under a sync/threaded runtime (a later blocking read just parks a thread),
//! but under an async runtime the same blocking read parks an executor worker and starves
//! the runtime, so the read must materialize eagerly instead.

use core::sync::atomic::{AtomicU8, Ordering};

/// How the host program is being driven.
///
/// Set once near program start with [`set_runtime_kind`]; read anywhere with
/// [`runtime_kind`]. Defaults to [`RuntimeKind::Sync`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeKind {
    /// Synchronous, thread-based runtime (the default).
    ///
    /// Blocking readbacks are fine here, so tensor reads may defer the device→host copy
    /// lazily and materialize it on first access.
    #[default]
    Sync = 0,
    /// Asynchronous runtime (e.g. tokio).
    ///
    /// Blocking a runtime worker starves the executor, so tensor reads must materialize
    /// eagerly inside the awaited future rather than deferring a blocking copy.
    Async = 1,
    /// Restricted no-std environment.
    NoStd = 2,
}

impl RuntimeKind {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => RuntimeKind::Async,
            2 => RuntimeKind::NoStd,
            _ => RuntimeKind::Sync,
        }
    }
}

static RUNTIME_KIND: AtomicU8 = AtomicU8::new(RuntimeKind::Sync as u8);

/// Declare the [kind](RuntimeKind) of runtime the host program is running on.
///
/// Intended to be called once near program start (e.g. when a remote server declares that
/// it hosts the backend on an async runtime). Backends read it via [`runtime_kind`].
pub fn set_runtime_kind(kind: RuntimeKind) {
    RUNTIME_KIND.store(kind as u8, Ordering::Relaxed);
}

/// Return the currently declared [kind](RuntimeKind) of runtime the host program runs on.
///
/// Defaults to [`RuntimeKind::Sync`] until [`set_runtime_kind`] is called.
pub fn runtime_kind() -> RuntimeKind {
    RuntimeKind::from_u8(RUNTIME_KIND.load(Ordering::Relaxed))
}
