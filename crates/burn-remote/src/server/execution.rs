use burn_backend::{
    ExecutionError, ExecutionMode, ExecutionModeToken, ExecutionStatistics, UnknownExecutionMode,
};
use burn_ir::BackendIr;
use burn_router::TensorInterpreter;
use burn_std::id::StreamId;

/// The execution modes one session has open on the server's backend.
///
/// A mode outlives no session: a client that disconnects with a mode open would otherwise leave
/// the server discarding every later launch, its other sessions' included, so the session
/// [closes](Self::close) what it still holds as it ends. A mode stays held until a close of it
/// succeeds, so a failed close is tried again then. A session reads and closes only the modes it
/// opened.
///
/// The session keeps the token its backend handed out, which names the server stream the mode
/// was opened on (see [`ServerStreams`](super::streams::ServerStreams)), where it closes. The
/// client's copy names the client's stream instead, so a mode is found by its token's id.
#[derive(Default)]
pub struct SessionExecutionModes {
    open: Vec<ExecutionModeToken>,
}

impl SessionExecutionModes {
    /// Open `mode` after everything the session ran on `stream`.
    pub fn open<B: BackendIr>(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
        mode: ExecutionMode,
    ) -> Result<Option<ExecutionModeToken>, ExecutionError> {
        let opened = stream.executes(|| runner.execution_mode_open(mode))?;
        self.open.extend(opened);
        Ok(opened)
    }

    /// What the work under the session's mode `token` has triggered so far.
    pub fn statistics<B: BackendIr>(
        &self,
        runner: &TensorInterpreter<B>,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        let held = self.held(token).ok_or(UnknownExecutionMode)?;
        runner.execution_mode_statistics(self.open[held])
    }

    /// Close the session's mode `token` after everything the session ran on the stream it was
    /// opened on.
    pub fn close<B: BackendIr>(
        &mut self,
        runner: &TensorInterpreter<B>,
        token: ExecutionModeToken,
    ) -> Result<ExecutionStatistics, ExecutionError> {
        let held = self.held(token).ok_or(UnknownExecutionMode)?;
        let token = self.open[held];
        let closed = token
            .opened_on
            .executes(|| runner.execution_mode_close(token))?;
        self.open.swap_remove(held);
        Ok(closed)
    }

    /// Where the session keeps the mode the client's `token` names, by its id.
    fn held(&self, token: ExecutionModeToken) -> Option<usize> {
        self.open.iter().position(|held| held.id == token.id)
    }

    /// Close every mode the session still holds, as it ends.
    pub fn close_all<B: BackendIr>(self, runner: &TensorInterpreter<B>) {
        for token in self.open {
            let closed = token
                .opened_on
                .executes(|| runner.execution_mode_close(token));
            if let Err(err) = closed {
                log::warn!("An execution mode left open by its session failed to close: {err}");
            }
        }
    }
}
