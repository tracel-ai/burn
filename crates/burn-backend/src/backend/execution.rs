//! What a device does with the work it is asked to run.
//!
//! A pass that only wants a workload's kernels compiled, or its autotune keys
//! tuned, does not want the workload's results: it runs the workload once under
//! a [`ProcessMode`] that discards the launches, and reads what the pass
//! triggered from its [`ExecutionStatistics`]. A step that has to run for real
//! inside such a pass puts its stream back to executing with a [`StreamMode`].
//!
//! The vocabulary is the backend's own rather than any runtime's, since
//! [`Backend`](super::Backend) is also implemented by backends that compile
//! nothing and run every launch; and it travels, so a backend executing on
//! another machine applies it there.

use burn_std::{ExecutionError, backtrace::BackTrace, id::StreamId};

/// What a device does with the work it is asked to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProcessMode {
    /// Launches run. What a device does when no mode is open on it.
    Execute,
    /// Launches are discarded. Their kernels are compiled and their autotune
    /// keys registered, but no candidate is measured and nothing is decided.
    CompileOnly,
    /// Launches are discarded. Their kernels are compiled, and their autotune
    /// keys tuned by measuring the candidates.
    CompileAndAutotune,
}

/// Whether a stream's launches run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StreamMode {
    /// They run.
    Execute,
    /// They are discarded, their kernels compiled as the open [`ProcessMode`]
    /// says.
    Discard,
}

/// A mode a device opens, with
/// [`execution_mode_open`](super::Backend::execution_mode_open).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExecutionMode {
    /// For all the device's work.
    Process(ProcessMode),
    /// For the calling stream's work alone, whatever the process mode says.
    Stream(StreamMode),
}

/// An open [`ExecutionMode`] on a device, handed out by
/// [`execution_mode_open`](super::Backend::execution_mode_open) and passed back
/// to read its statistics and to close it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExecutionModeToken {
    /// The backend's own identifier for the open mode.
    pub id: u64,
    /// The stream the mode was opened on, which is where it closes: the work
    /// registered there before the close runs under the mode, whichever
    /// thread closes it.
    pub opened_on: StreamId,
}

/// What the work run under an open [`ProcessMode`] triggered: the kernels
/// obtained and the autotune keys tuned. Every count only grows.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ExecutionStatistics {
    /// The kernels obtained.
    pub compilation: CompilationStatistics,
    /// The autotune keys tuned.
    pub autotune: AutotuneStatistics,
}

/// The kernels the launches under an open [`ProcessMode`] obtained.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CompilationStatistics {
    /// Asked for by a launch that found them neither compiled nor queued. A
    /// kernel tried again after failing registers again.
    pub registered: u64,
    /// Compiled by the backend's compiler.
    pub compiled: u64,
    /// Read from the compilation store instead.
    pub loaded: u64,
    /// Failed to compile.
    pub failed: u64,
    /// Dropped before it settled: its compilation panicked, or the device
    /// holding it went away.
    pub abandoned: u64,
    /// Compiled or read from the store, then refused by the device when it
    /// was loaded: counted among [`compiled`](Self::compiled) or
    /// [`loaded`](Self::loaded) too.
    pub refused: u64,
    /// Kept in the compilation store once loaded: what the store grew by.
    pub stored: u64,
}

impl CompilationStatistics {
    /// The kernels done, however they ended: never more than
    /// [`registered`](Self::registered).
    pub fn settled(&self) -> u64 {
        self.compiled + self.loaded + self.failed + self.abandoned
    }
}

/// The autotune keys tuned under an open [`ProcessMode`]. A key with one
/// candidate is answered, not tuned, and counts nowhere.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct AutotuneStatistics {
    /// Registered by a [`CompileOnly`](ProcessMode::CompileOnly) pass, or
    /// reached unregistered by a tune that measures or by a read of the
    /// persistent autotune cache.
    pub registered: u64,
    /// Tuned by measuring their candidates.
    pub measured: u64,
    /// Read from the persistent autotune cache instead of measured.
    pub loaded: u64,
    /// Every candidate failed: the pick was made unmeasured.
    pub failed: u64,
    /// Dropped before it settled.
    pub abandoned: u64,
    /// Measured, and kept in the persistent autotune cache.
    pub persisted: u64,
}

impl AutotuneStatistics {
    /// The keys done, however they ended: never more than
    /// [`registered`](Self::registered).
    pub fn settled(&self) -> u64 {
        self.measured + self.loaded + self.failed + self.abandoned
    }
}

/// A backend has no mode open under the token it was asked about: it never
/// handed it out, it is closed, or another session of a remote device opened
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownExecutionMode;

impl From<UnknownExecutionMode> for ExecutionError {
    fn from(_: UnknownExecutionMode) -> Self {
        ExecutionError::Generic {
            reason: alloc::string::String::from(
                "no execution mode is open under this token on this backend",
            ),
            backtrace: BackTrace::capture(),
        }
    }
}
