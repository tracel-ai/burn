use burn_backend::{
    ExecutionError, ExecutionStatistics, ProcessMode, ProcessModeToken, StreamMode,
    StreamModeToken, execution_mode_unknown,
};
use burn_ir::BackendIr;
use burn_router::TensorInterpreter;
use burn_std::id::StreamId;

/// The execution modes one session has open on the server's backend.
///
/// A mode outlives no session: a client that disconnects with a mode open would otherwise leave
/// the server discarding every later launch, its other sessions' included, so the session
/// [closes](Self::close) what it still holds as it ends. A session reads and closes only the
/// modes it opened.
///
/// Every stream here is a server stream (see [`ServerStreams`](super::streams::ServerStreams)).
#[derive(Default)]
pub struct SessionExecutionModes {
    process: Vec<ProcessModeToken>,
    /// Each with the stream it was opened on, which is where it closes.
    streams: Vec<(StreamModeToken, StreamId)>,
}

impl SessionExecutionModes {
    /// Run the backend's work under `mode` after everything the session ran on `stream`.
    pub fn open_process<B: BackendIr>(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
        mode: ProcessMode,
    ) -> Result<Option<ProcessModeToken>, ExecutionError> {
        let opened = stream.executes(|| runner.process_mode_open(mode))?;
        self.process.extend(opened);
        Ok(opened)
    }

    /// What the work under the session's mode `token` has triggered so far.
    pub fn process_statistics<B: BackendIr>(
        &self,
        runner: &TensorInterpreter<B>,
        token: ProcessModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        match self.process.contains(&token) {
            true => runner.process_mode_statistics(token),
            false => Err(execution_mode_unknown()),
        }
    }

    /// Close the session's mode `token` after everything the session ran on `stream`.
    pub fn close_process<B: BackendIr>(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
        token: ProcessModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        let Some(position) = self.process.iter().position(|held| *held == token) else {
            return Err(execution_mode_unknown());
        };
        self.process.swap_remove(position);
        stream.executes(|| runner.process_mode_close(token))
    }

    /// Put `stream` in `mode` after everything the session ran on it.
    pub fn open_stream<B: BackendIr>(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
        mode: StreamMode,
    ) -> Result<Option<StreamModeToken>, ExecutionError> {
        let opened = stream.executes(|| runner.stream_mode_open(mode))?;
        self.streams.extend(opened.map(|token| (token, stream)));
        Ok(opened)
    }

    /// Close the session's stream mode `token` on the stream it was opened on.
    pub fn close_stream<B: BackendIr>(
        &mut self,
        runner: &TensorInterpreter<B>,
        token: StreamModeToken,
    ) -> Result<(), ExecutionError> {
        let Some(position) = self.streams.iter().position(|(held, _)| *held == token) else {
            return Err(execution_mode_unknown());
        };
        let (token, stream) = self.streams.swap_remove(position);
        stream.executes(|| runner.stream_mode_close(token))
    }

    /// Close every mode the session still holds, as it ends.
    pub fn close<B: BackendIr>(self, runner: &TensorInterpreter<B>) {
        for (token, stream) in self.streams {
            if let Err(err) = stream.executes(|| runner.stream_mode_close(token)) {
                log::warn!("A stream mode left open by its session failed to close: {err}");
            }
        }
        for token in self.process {
            if let Err(err) = runner.process_mode_close(token) {
                log::warn!("A process mode left open by its session failed to close: {err}");
            }
        }
    }
}
