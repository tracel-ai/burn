pub use burn_backend::{
    AutotuneStatistics, CompilationStatistics, ExecutionStatistics, ProcessMode, StreamMode,
};

use crate::Device;
use burn_backend::{Backend, ExecutionError, ProcessModeToken, StreamModeToken};
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
    device: Device,
    mode: ProcessMode,
    /// `None` on a backend that runs every launch whatever the mode, and once
    /// closed.
    token: Option<ProcessModeToken>,
}

impl ProcessModeGuard {
    /// Open `mode` on `device`, after every operation registered so far on
    /// the calling stream.
    ///
    /// # Errors
    ///
    /// Another mode is open where this one would apply, or the device could
    /// not be reached.
    pub fn new(device: Device, mode: ProcessMode) -> Result<Self, ExecutionError> {
        let token = Dispatch::process_mode_open(device.as_dispatch(), mode)?;
        Ok(Self {
            device,
            mode,
            token,
        })
    }

    /// The mode it holds open.
    pub fn mode(&self) -> ProcessMode {
        self.mode
    }

    /// Whether the device applies the mode. A backend that compiles nothing
    /// runs every launch whatever the mode, and counts nothing.
    pub fn applies(&self) -> bool {
        self.token.is_some()
    }

    /// What reads what the work under the mode triggered, from any thread.
    pub fn reader(&self) -> ProcessModeReader {
        ProcessModeReader {
            device: self.device.clone(),
            token: self.token,
        }
    }

    /// Close the mode, after every operation registered so far on the
    /// calling stream, and answer what the work under it triggered.
    ///
    /// # Errors
    ///
    /// The device could not be reached.
    pub fn close(mut self) -> Result<ExecutionStatistics, ExecutionError> {
        match self.token.take() {
            Some(token) => Dispatch::process_mode_close(self.device.as_dispatch(), token),
            None => Ok(ExecutionStatistics::default()),
        }
    }
}

impl Drop for ProcessModeGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            let _ = Dispatch::process_mode_close(self.device.as_dispatch(), token);
        }
    }
}

/// Reads what the work under an open [`ProcessModeGuard`] triggered, from any
/// thread, and opens nothing.
#[derive(Debug, Clone)]
pub struct ProcessModeReader {
    device: Device,
    token: Option<ProcessModeToken>,
}

impl ProcessModeReader {
    /// What the work under the mode triggered so far. Nothing, on a backend
    /// that does not apply it.
    ///
    /// # Errors
    ///
    /// The mode is closed, or the device could not be reached.
    pub fn statistics(&self) -> Result<ExecutionStatistics, ExecutionError> {
        match self.token {
            Some(token) => Dispatch::process_mode_statistics(self.device.as_dispatch(), token),
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
    device: Device,
    /// `None` on a backend that discards nothing, and once closed.
    token: Option<StreamModeToken>,
}

impl StreamModeGuard {
    /// Put the calling stream of `device` in `mode`, after every operation
    /// registered so far on it.
    ///
    /// # Errors
    ///
    /// The device could not be reached.
    pub fn new(device: Device, mode: StreamMode) -> Result<Self, ExecutionError> {
        let token = Dispatch::stream_mode_open(device.as_dispatch(), mode)?;
        Ok(Self { device, token })
    }

    /// Close the mode, after every operation registered so far on the stream
    /// it was opened on.
    ///
    /// # Errors
    ///
    /// The device could not be reached.
    pub fn close(mut self) -> Result<(), ExecutionError> {
        match self.token.take() {
            Some(token) => Dispatch::stream_mode_close(self.device.as_dispatch(), token),
            None => Ok(()),
        }
    }
}

impl Drop for StreamModeGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            let _ = Dispatch::stream_mode_close(self.device.as_dispatch(), token);
        }
    }
}
