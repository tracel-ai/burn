pub use burn_backend::{
    AutotuneStatistics, CompilationStatistics, ExecutionMode, ExecutionStatistics, ProcessMode,
    StreamMode,
};

use crate::Device;
use burn_backend::{Backend, ExecutionError, ExecutionModeToken};
use burn_dispatch::Dispatch;

/// An [`ExecutionMode`] open on a device, from [`Device::mode`] until it is
/// [closed](Self::close) or dropped.
///
/// The mode applies where the device's work runs, so a device whose backend
/// executes on another machine runs under it there. The guard holds the mode
/// until a close of it succeeds: a mode left open would discard every later
/// launch, so dropping the guard closes it as [`close`](Self::close) does —
/// trying again after a close that failed — minus the answer.
#[derive(Debug)]
#[must_use = "the mode closes when the guard drops"]
pub struct ExecutionModeGuard {
    device: Device,
    mode: ExecutionMode,
    /// `None` on a backend that does not apply the mode, and once closed.
    token: Option<ExecutionModeToken>,
}

impl ExecutionModeGuard {
    pub(crate) fn new(device: Device, mode: ExecutionMode) -> Result<Self, ExecutionError> {
        let token = Dispatch::execution_mode_open(device.as_dispatch(), mode)?;
        Ok(Self {
            device,
            mode,
            token,
        })
    }

    /// The mode it holds open.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Whether the device applies the mode. A backend that compiles nothing
    /// runs every launch whatever the mode, and counts nothing.
    pub fn applies(&self) -> bool {
        self.token.is_some()
    }

    /// What reads what the work under the mode triggered, from any thread.
    /// Nothing, for a [stream mode](ExecutionMode::Stream).
    pub fn reader(&self) -> ExecutionStatisticsReader {
        ExecutionStatisticsReader {
            device: self.device.clone(),
            token: self.token,
        }
    }

    /// Close the mode, after every operation registered so far on the stream
    /// that opened it, and answer what the work under it triggered.
    ///
    /// # Errors
    ///
    /// The work queued before the close failed, or the device could not be
    /// reached. Dropping the guard then tries the close again.
    pub fn close(mut self) -> Result<ExecutionStatistics, ExecutionError> {
        self.try_close()
    }

    fn try_close(&mut self) -> Result<ExecutionStatistics, ExecutionError> {
        let Some(token) = self.token else {
            return Ok(ExecutionStatistics::default());
        };
        let closed = Dispatch::execution_mode_close(self.device.as_dispatch(), token)?;
        self.token = None;
        Ok(closed)
    }
}

impl Drop for ExecutionModeGuard {
    fn drop(&mut self) {
        // What failed is the answer of `close`, which a caller wanting it calls.
        let _ = self.try_close();
    }
}

/// Reads what the work under an open [`ExecutionModeGuard`] triggered, from
/// any thread, and opens nothing.
#[derive(Debug, Clone)]
pub struct ExecutionStatisticsReader {
    device: Device,
    token: Option<ExecutionModeToken>,
}

impl ExecutionStatisticsReader {
    /// What the work under the mode triggered so far. Nothing, on a backend
    /// that does not apply it.
    ///
    /// # Errors
    ///
    /// The mode is closed, or the device could not be reached.
    pub fn statistics(&self) -> Result<ExecutionStatistics, ExecutionError> {
        match self.token {
            Some(token) => Dispatch::execution_mode_statistics(self.device.as_dispatch(), token),
            None => Ok(ExecutionStatistics::default()),
        }
    }
}
