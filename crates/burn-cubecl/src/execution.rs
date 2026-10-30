//! The execution modes a [`CubeBackend`](crate::CubeBackend) device opens, on
//! cubecl's.
//!
//! Cubecl applies a process mode to the whole process, so every device of
//! this process runs under the one open here, and one mode is open at a time.
//! Opens of the mode already open join it: they share its collector, so each
//! token reads what every device triggered while the mode was open.

use alloc::vec::Vec;
use burn_backend::{
    AutotuneStatistics, CompilationStatistics, ExecutionError, ExecutionStatistics, ProcessMode,
    ProcessModeToken, StreamMode, StreamModeToken, execution_mode_unknown,
};
use burn_std::sync::Mutex;
use cubecl::client::Client;
use cubecl::execution as cube;

/// The process mode open in this process, and the tokens holding it open.
struct OpenProcessMode {
    mode: ProcessMode,
    /// One override per token: cubecl closes the mode once the last drops.
    /// Before the collector they count into, so they drop first.
    overrides: Vec<(ProcessModeToken, cube::ProcessModeOverride)>,
    collector: cube::StatisticsCollector,
}

/// The modes this backend has open in this process, by the tokens it handed
/// out for them.
pub(crate) struct OpenModes {
    process: Option<OpenProcessMode>,
    streams: Vec<(StreamModeToken, cube::StreamModeOverride)>,
    next_id: u64,
}

/// The process's one record of what is open: cubecl's modes are process-wide.
pub(crate) static OPEN_MODES: Mutex<OpenModes> = Mutex::new(OpenModes {
    process: None,
    streams: Vec::new(),
    next_id: 0,
});

impl OpenModes {
    /// Open `mode` for the whole process, or join it if it is already open.
    pub(crate) fn open_process(
        &mut self,
        mode: ProcessMode,
    ) -> Result<ProcessModeToken, ExecutionError> {
        let token = ProcessModeToken { id: self.next_id() };
        let process = match &mut self.process {
            Some(process) if process.mode != mode => {
                return Err(ExecutionError::with_context(alloc::format!(
                    "{mode:?} cannot open while {:?} is open: the cubecl runtimes apply one \
                     process mode to the whole process",
                    process.mode
                )));
            }
            Some(process) => process,
            None => self.process.insert(OpenProcessMode {
                mode,
                overrides: Vec::new(),
                collector: cube::StatisticsCollector::new(),
            }),
        };
        let execution = cube::ProcessModeOverride::new(mode.into_cube(), &process.collector);
        process.overrides.push((token, execution));
        Ok(token)
    }

    /// What the work under the mode `token` holds open has triggered so far.
    pub(crate) fn process_statistics(
        &self,
        token: ProcessModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        match &self.process {
            Some(process) if process.holds(token) => Ok(process.statistics()),
            _ => Err(execution_mode_unknown()),
        }
    }

    /// Let go of the mode `token` holds open, closing it if no other token
    /// does.
    pub(crate) fn close_process(
        &mut self,
        token: ProcessModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        let Some(process) = self.process.as_mut().filter(|process| process.holds(token)) else {
            return Err(execution_mode_unknown());
        };
        process.overrides.retain(|(held, _)| *held != token);
        let statistics = process.statistics();
        if process.overrides.is_empty() {
            self.process = None;
        }
        Ok(statistics)
    }

    /// Put `client`'s stream in `mode` until the token closes.
    pub(crate) fn open_stream(&mut self, mode: StreamMode, client: &Client) -> StreamModeToken {
        let token = StreamModeToken { id: self.next_id() };
        let execution = cube::StreamModeOverride::new(mode.into_cube(), client);
        self.streams.push((token, execution));
        token
    }

    /// Close the stream mode `token` holds open.
    pub(crate) fn close_stream(&mut self, token: StreamModeToken) -> Result<(), ExecutionError> {
        let before = self.streams.len();
        self.streams.retain(|(held, _)| *held != token);
        match self.streams.len() < before {
            true => Ok(()),
            false => Err(execution_mode_unknown()),
        }
    }

    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
}

impl OpenProcessMode {
    fn holds(&self, token: ProcessModeToken) -> bool {
        self.overrides.iter().any(|(held, _)| *held == token)
    }

    fn statistics(&self) -> ExecutionStatistics {
        let statistics = self.collector.statistics();
        let compilation = statistics.compilation;
        let autotune = statistics.autotune;
        ExecutionStatistics {
            compilation: CompilationStatistics {
                registered: compilation.registered as u64,
                compiled: compilation.compiled as u64,
                loaded: compilation.loaded as u64,
                failed: compilation.failed as u64,
                abandoned: compilation.abandoned as u64,
                refused: compilation.refused as u64,
                stored: compilation.stored as u64,
            },
            autotune: AutotuneStatistics {
                registered: autotune.registered as u64,
                measured: autotune.measured as u64,
                loaded: autotune.loaded as u64,
                failed: autotune.failed as u64,
                abandoned: autotune.abandoned as u64,
                persisted: autotune.persisted as u64,
            },
        }
    }
}

/// The cubecl name of a mode.
trait IntoCube {
    type Cube;

    fn into_cube(self) -> Self::Cube;
}

impl IntoCube for ProcessMode {
    type Cube = cube::ProcessMode;

    fn into_cube(self) -> cube::ProcessMode {
        match self {
            Self::Execute => cube::ProcessMode::Execute,
            Self::CompileOnly => cube::ProcessMode::CompileOnly,
            Self::CompileAndAutotune => cube::ProcessMode::CompileAndAutotune,
        }
    }
}

impl IntoCube for StreamMode {
    type Cube = cube::StreamMode;

    fn into_cube(self) -> cube::StreamMode {
        match self {
            Self::Execute => cube::StreamMode::Execute,
            Self::Discard => cube::StreamMode::Discard,
        }
    }
}
