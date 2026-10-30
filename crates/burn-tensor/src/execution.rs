pub use burn_backend::{
    AutotuneStatistics, CompilationStatistics, ExecutionStatistics, ProcessMode, StreamMode,
};

use crate::Device;
use burn_backend::{Backend, ExecutionError, ExecutionMode, ExecutionModeToken};
use burn_dispatch::Dispatch;

/// A [`ProcessMode`] open on a device, from
/// [`Device::process_mode`](crate::Device::process_mode) until it is
/// [closed](Self::close) or dropped.
///
/// The mode applies where the device's work runs, so a device whose backend
/// executes on another machine runs under it there. Dropping the guard closes
/// it as [`close`](Self::close) does, minus the answer: a close that failed is
/// only learned through [`close`](Self::close).
#[derive(Debug)]
#[must_use = "the mode closes when the guard drops"]
pub struct ProcessModeGuard {
    mode: ProcessMode,
    open: OpenExecutionMode,
}

impl ProcessModeGuard {
    pub(crate) fn new(device: Device, mode: ProcessMode) -> Result<Self, ExecutionError> {
        Ok(Self {
            mode,
            open: OpenExecutionMode::new(device, ExecutionMode::Process(mode))?,
        })
    }

    /// The mode it holds open.
    pub fn mode(&self) -> ProcessMode {
        self.mode
    }

    /// Whether the device applies the mode. A backend that compiles nothing
    /// runs every launch whatever the mode, and counts nothing.
    pub fn applies(&self) -> bool {
        self.open.token.is_some()
    }

    /// What reads what the work under the mode triggered, from any thread.
    pub fn reader(&self) -> ExecutionStatisticsReader {
        ExecutionStatisticsReader {
            device: self.open.device.clone(),
            token: self.open.token,
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
        self.open.close()
    }
}

/// Reads what the work under an open [`ProcessModeGuard`] triggered, from any
/// thread, and opens nothing.
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

/// A [`StreamMode`] open on the stream that opened it, from
/// [`Device::stream_mode`](crate::Device::stream_mode) until it is
/// [closed](Self::close) or dropped.
#[derive(Debug)]
#[must_use = "the mode closes when the guard drops"]
pub struct StreamModeGuard {
    open: OpenExecutionMode,
}

impl StreamModeGuard {
    pub(crate) fn new(device: Device, mode: StreamMode) -> Result<Self, ExecutionError> {
        Ok(Self {
            open: OpenExecutionMode::new(device, ExecutionMode::Stream(mode))?,
        })
    }

    /// Close the mode, after every operation registered so far on the stream
    /// it was opened on.
    ///
    /// # Errors
    ///
    /// The work queued before the close failed, or the device could not be
    /// reached. Dropping the guard then tries the close again.
    pub fn close(mut self) -> Result<(), ExecutionError> {
        self.open.close().map(|_| ())
    }
}

/// A mode open on a device, held until a close of it succeeds: a mode left
/// open would discard every later launch, so a close that failed is tried
/// again on drop.
#[derive(Debug)]
struct OpenExecutionMode {
    device: Device,
    /// `None` on a backend that does not apply the mode, and once closed.
    token: Option<ExecutionModeToken>,
}

impl OpenExecutionMode {
    fn new(device: Device, mode: ExecutionMode) -> Result<Self, ExecutionError> {
        let token = Dispatch::execution_mode_open(device.as_dispatch(), mode)?;
        Ok(Self { device, token })
    }

    fn close(&mut self) -> Result<ExecutionStatistics, ExecutionError> {
        let Some(token) = self.token else {
            return Ok(ExecutionStatistics::default());
        };
        let closed = Dispatch::execution_mode_close(self.device.as_dispatch(), token)?;
        self.token = None;
        Ok(closed)
    }
}

impl Drop for OpenExecutionMode {
    fn drop(&mut self) {
        // What failed is the answer of `close`, which a caller wanting it calls.
        let _ = self.close();
    }
}
