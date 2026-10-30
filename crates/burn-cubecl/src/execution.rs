//! The execution modes a [`CubeBackend`](crate::CubeBackend) device opens, on
//! cubecl's.
//!
//! Cubecl applies a process mode to the whole process, so every device of
//! this process runs under the one open here, and one is open at a time.
//! Opens of the process mode already open join it: they share its collector,
//! so each token reads what every device triggered while the mode was open.

use alloc::vec::Vec;
use burn_backend::{
    ExecutionError, ExecutionMode, ExecutionModeToken, ExecutionStatistics, ProcessMode,
    UnknownExecutionMode,
};
use burn_std::{id::StreamId, sync::Mutex};
use cubecl::client::Client;
use cubecl::execution as cube;

/// The process mode open in this process, and the tokens holding it open.
struct OpenProcessMode {
    mode: ProcessMode,
    /// One override per token: cubecl closes the mode once the last drops.
    /// Before the collector they count into, so they drop first.
    overrides: Vec<(ExecutionModeToken, cube::ProcessModeOverride)>,
    collector: cube::StatisticsCollector,
}

/// The modes this backend has open in this process, by the tokens it handed
/// out for them.
pub(crate) struct OpenModes {
    process: Option<OpenProcessMode>,
    streams: Vec<(ExecutionModeToken, cube::StreamModeOverride)>,
    next_id: u64,
}

/// The process's one record of what is open: cubecl's modes are process-wide.
pub(crate) static OPEN_MODES: Mutex<OpenModes> = Mutex::new(OpenModes {
    process: None,
    streams: Vec::new(),
    next_id: 0,
});

impl OpenModes {
    /// Open `mode`: a process mode for the whole process, joining it if it is
    /// already open, or a stream mode for `client`'s stream.
    pub(crate) fn open(
        &mut self,
        mode: ExecutionMode,
        client: &Client,
    ) -> Result<ExecutionModeToken, ExecutionError> {
        let token = ExecutionModeToken {
            id: self.next_id(),
            opened_on: StreamId::current(),
        };
        match mode {
            ExecutionMode::Process(mode) => self.open_process(token, mode)?,
            ExecutionMode::Stream(mode) => {
                let execution = cube::StreamModeOverride::new(mode.into(), client);
                self.streams.push((token, execution));
            }
        }
        Ok(token)
    }

    /// What the work under the mode `token` holds open has triggered so far.
    pub(crate) fn statistics(
        &self,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        if self.streams.iter().any(|(held, _)| *held == token) {
            return Ok(ExecutionStatistics::default());
        }
        match &self.process {
            Some(process) if process.holds(token) => Ok(process.collector.statistics().into()),
            _ => Err(UnknownExecutionMode.into()),
        }
    }

    /// Let go of the mode `token` holds open. A process mode closes once no
    /// other token holds it.
    pub(crate) fn close(
        &mut self,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        if let Some(position) = self.streams.iter().position(|(held, _)| *held == token) {
            self.streams.swap_remove(position);
            return Ok(ExecutionStatistics::default());
        }
        let Some(process) = self.process.as_mut().filter(|process| process.holds(token)) else {
            return Err(UnknownExecutionMode.into());
        };
        process.overrides.retain(|(held, _)| *held != token);
        let statistics = process.collector.statistics().into();
        if process.overrides.is_empty() {
            self.process = None;
        }
        Ok(statistics)
    }

    /// Cubecl's override is built before anything is recorded, so one that
    /// panics leaves no mode behind that nothing could close.
    fn open_process(
        &mut self,
        token: ExecutionModeToken,
        mode: ProcessMode,
    ) -> Result<(), ExecutionError> {
        match &mut self.process {
            Some(process) if process.mode != mode => {
                Err(ExecutionError::with_context(alloc::format!(
                    "{mode:?} cannot open while {:?} is open: the cubecl runtimes apply one \
                     process mode to the whole process",
                    process.mode
                )))
            }
            Some(process) => {
                let execution = cube::ProcessModeOverride::new(mode.into(), &process.collector);
                process.overrides.push((token, execution));
                Ok(())
            }
            None => {
                let collector = cube::StatisticsCollector::new();
                let execution = cube::ProcessModeOverride::new(mode.into(), &collector);
                self.process = Some(OpenProcessMode {
                    mode,
                    overrides: alloc::vec![(token, execution)],
                    collector,
                });
                Ok(())
            }
        }
    }

    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
}

impl OpenProcessMode {
    fn holds(&self, token: ExecutionModeToken) -> bool {
        self.overrides.iter().any(|(held, _)| *held == token)
    }
}
