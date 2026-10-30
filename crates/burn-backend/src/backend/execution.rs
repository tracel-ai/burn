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

use burn_std::{ExecutionError, backtrace::BackTrace};

/// What a device does with the work it is asked to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProcessMode {
    /// Launches run. What a device does when no mode is open on it.
    Execute,
    /// Launches and tune candidates queue their kernels and are discarded. The
    /// queue compiles in one batch, at the next launch that loads a kernel. A
    /// tune measures and decides nothing.
    CompileOnly,
    /// Launches compile their kernels and are discarded. A tune measures its
    /// candidates for real.
    CompileAndAutotune,
}

impl ProcessMode {
    /// The mode every stream has under it, unless a stream opens its own.
    pub fn stream_mode(self) -> StreamMode {
        match self {
            Self::Execute => StreamMode::Execute,
            Self::CompileOnly | Self::CompileAndAutotune => StreamMode::Discard,
        }
    }
}

/// Whether a stream's launches run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StreamMode {
    /// They run.
    Execute,
    /// Their kernels compile, as the open [`ProcessMode`] says, and the
    /// launches are discarded.
    Discard,
}

/// An open [`ProcessMode`] on a device, handed out by
/// [`process_mode_open`](super::Backend::process_mode_open) and passed back to
/// read its statistics and to close it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProcessModeToken {
    /// The backend's own identifier for the open mode.
    pub id: u64,
}

/// An open [`StreamMode`] on one stream of a device, handed out by
/// [`stream_mode_open`](super::Backend::stream_mode_open) and passed back to
/// close it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StreamModeToken {
    /// The backend's own identifier for the open mode.
    pub id: u64,
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
    /// Queued, or asked for by a launch that found them neither loaded nor
    /// queued. A kernel tried again after failing registers again.
    pub registered: u64,
    /// Compiled by the backend's compiler.
    pub compiled: u64,
    /// Read from the compilation store instead.
    pub loaded: u64,
    /// Failed to compile.
    pub failed: u64,
    /// Dropped before it settled: the batch compiling it panicked, or the
    /// device holding it went away.
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
    /// Gathered by a [`CompileOnly`](ProcessMode::CompileOnly) pass, or
    /// reached ungathered by a tune that measures or a read of the persistent
    /// autotune cache.
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

/// The error a backend answers with when asked about a mode it never handed
/// out, which is every mode for a backend that opens none.
pub fn execution_mode_unknown() -> ExecutionError {
    ExecutionError::Generic {
        reason: alloc::string::String::from(
            "no execution mode is open under this token on this backend",
        ),
        backtrace: BackTrace::capture(),
    }
}
