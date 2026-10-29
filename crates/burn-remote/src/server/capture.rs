use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use burn_backend::{BackendGraph, ExecutionError};
use burn_ir::{BackendIr, CapturedGraphId};
use burn_router::TensorInterpreter;
use burn_std::id::StreamId;

use crate::shared::SessionId;

/// The session holding the graph capture of one device of the server, if any.
///
/// A backend keeps its capture state per backend stream, and logical streams fold onto backend
/// streams, so a second session capturing on the device could record into the first one's
/// graph or, under CUDA's global capture mode, invalidate it. The server therefore runs one
/// capture per device: a session claims the device from the capture's preparation until it
/// ends, and any other session's capture is refused meanwhile, so its client falls back to
/// re-running the closure.
///
/// The claim does not stop the other sessions' operations. One that folds onto the capturing
/// backend stream pays what the backend charges a neighbour of a capture: a refused read on
/// wgpu; on CUDA and HIP a deferred flush, and launches that join the recording. Keeping them
/// out takes a backend stream reserved for the capture, which the backend does not offer.
///
/// A session that prepares a capture and never starts it holds the device until it ends.
#[derive(Clone, Default)]
pub(crate) struct CaptureClaim {
    holder: Arc<Mutex<Option<SessionId>>>,
}

impl CaptureClaim {
    /// Claim the device for `session`'s capture, unless another session holds it.
    fn claim(&self, session: SessionId) -> Result<(), ExecutionError> {
        let mut holder = self.holder.lock().unwrap_or_else(PoisonError::into_inner);
        match *holder {
            Some(other) if other != session => Err(ExecutionError::with_context(format!(
                "the server's device is capturing a graph for session {other}; one capture runs \
                 per device"
            ))),
            _ => {
                *holder = Some(session);
                Ok(())
            }
        }
    }

    /// Let go of the device, if `session` holds it.
    fn release(&self, session: SessionId) {
        let mut holder = self.holder.lock().unwrap_or_else(PoisonError::into_inner);
        if *holder == Some(session) {
            *holder = None;
        }
    }
}

/// The graph captures of one session: the one it has open, the graphs it keeps for its client
/// to replay, and the replays that failed since the client last read.
///
/// Every stream here is a server stream (see [`ServerStreams`](super::streams::ServerStreams)).
/// The session holds its device's [claim](CaptureClaim) exactly while a capture is open.
/// [Closing](Self::close) ends the open capture and releases the claim; dropping releases it
/// too, so a teardown that panics before or inside the close still hands the device on, though
/// its backend stream may be left recording and the next capture on it fail.
pub(crate) struct SessionCaptures<B: BackendIr> {
    session: SessionId,
    device: CaptureClaim,
    open: Option<OpenCapture>,
    /// The hardware graphs the backend recorded for the client, by the id the client picked (see
    /// [`Task::GraphStopCapture`](crate::shared::Task::GraphStopCapture)), until it releases them.
    graphs: HashMap<CapturedGraphId, BackendGraph<B>>,
    /// The first failed replay of each stream, until a read or a sync on that stream reports it.
    replay_failures: HashMap<StreamId, ExecutionError>,
}

/// A capture the session opened and has not ended, with the stream it runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenCapture {
    /// Prepared: the client is running its warmup before the recording starts.
    Prepared(StreamId),
    /// The backend records the stream's launches into a graph.
    Recording(StreamId),
}

impl OpenCapture {
    /// Why another capture cannot open beside this one.
    fn refusal(&self) -> ExecutionError {
        let (Self::Prepared(stream) | Self::Recording(stream)) = self;
        ExecutionError::with_context(format!(
            "the session already has a graph capture open on {stream}"
        ))
    }
}

impl<B: BackendIr> SessionCaptures<B> {
    /// The captures of `session`, which runs on the device `device` guards.
    pub(crate) fn new(session: SessionId, device: CaptureClaim) -> Self {
        Self {
            session,
            device,
            open: None,
            graphs: HashMap::new(),
            replay_failures: HashMap::new(),
        }
    }

    /// Prepare the backend for a capture on `stream`, claiming the device.
    pub(crate) fn prepare(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
    ) -> Result<(), ExecutionError> {
        if let Some(open) = self.open {
            return Err(open.refusal());
        }
        self.device.claim(self.session)?;
        let prepared = stream.executes(|| runner.graph_prepare());
        self.set_open(prepared.is_ok().then_some(OpenCapture::Prepared(stream)));
        prepared
    }

    /// Start recording `stream`'s launches, after the preparation on it if there was one.
    pub(crate) fn start(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
    ) -> Result<(), ExecutionError> {
        match self.open {
            Some(OpenCapture::Prepared(prepared)) if prepared == stream => {}
            Some(open) => return Err(open.refusal()),
            None => self.device.claim(self.session)?,
        }
        // A backend whose recording fails to start leaves its stream out of capture.
        let started = stream.executes(|| runner.graph_start_capture());
        self.set_open(started.is_ok().then_some(OpenCapture::Recording(stream)));
        started
    }

    /// Stop the recording on `stream` and keep its graph under `graph`.
    pub(crate) fn stop(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
        graph: CapturedGraphId,
    ) -> Result<(), ExecutionError> {
        if self.open != Some(OpenCapture::Recording(stream)) {
            return Err(ExecutionError::with_context(format!(
                "no graph capture is recording on {stream}"
            )));
        }
        let captured = self.end_recording(runner, stream)?;
        self.graphs.insert(graph, captured);
        Ok(())
    }

    /// Replay `graph` on `stream`. The client does not wait on the replay, so a failure is kept
    /// for the next read or sync of `stream`; a later replay that succeeds does not clear it, as
    /// it ran on what the failed one left.
    pub(crate) fn replay(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
        graph: CapturedGraphId,
    ) {
        let replayed = match self.graphs.get(&graph) {
            // Safety: a server's backend keeps the memory a graph's launches touched in place for
            // as long as the graph lives (see `BackendServer`), so a tensor the client dropped
            // since is never freed or reused under a replay. Holding the tensors here instead
            // would raise their reference counts and stop the client's in-place input refreshes
            // from reaching the graph. This replay runs in the order the client sent it, on the
            // stream it names.
            Some(captured) => stream.executes(|| unsafe { runner.graph_replay(captured) }),
            None => Err(ExecutionError::with_context(format!(
                "replay of unknown captured graph {graph:?}"
            ))),
        };
        if let Err(failure) = replayed {
            log::warn!("Session {}: {failure}", self.session);
            self.replay_failures.entry(stream).or_insert(failure);
        }
    }

    /// Drop `graph`; the client never replays it again.
    pub(crate) fn release(&mut self, graph: CapturedGraphId) {
        self.graphs.remove(&graph);
    }

    /// The first replay on `stream` that failed since the last call, which a read or sync of
    /// `stream` answers with in place of its own result.
    pub(crate) fn take_replay_failure(&mut self, stream: StreamId) -> Option<ExecutionError> {
        self.replay_failures.remove(&stream)
    }

    /// End the capture the client left open, so its stream leaves capture mode, and drop the
    /// graphs nothing replays any more.
    ///
    /// A prepared capture is started and stopped at once, since a backend leaves a prepared
    /// stream only through a recording.
    pub(crate) fn close(mut self, runner: &TensorInterpreter<B>) {
        let ended = match self.open {
            Some(OpenCapture::Prepared(stream)) => self
                .start(runner, stream)
                .and_then(|()| self.end_recording(runner, stream).map(drop)),
            Some(OpenCapture::Recording(stream)) => self.end_recording(runner, stream).map(drop),
            None => Ok(()),
        };
        if let Err(err) = ended {
            log::warn!(
                "Ending session {}'s open graph capture failed: {err}",
                self.session
            );
        }
    }

    /// Stop the open recording on `stream`, which leaves capture mode however the stop goes.
    fn end_recording(
        &mut self,
        runner: &TensorInterpreter<B>,
        stream: StreamId,
    ) -> Result<BackendGraph<B>, ExecutionError> {
        let stopped = stream.executes(|| runner.graph_stop_capture());
        self.set_open(None);
        stopped
    }

    /// Record `open` as the session's open capture, holding the device's claim exactly while
    /// there is one.
    fn set_open(&mut self, open: Option<OpenCapture>) {
        if open.is_none() {
            self.device.release(self.session);
        }
        self.open = open;
    }
}

impl<B: BackendIr> Drop for SessionCaptures<B> {
    fn drop(&mut self) {
        self.device.release(self.session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    /// A session's captures let go of the device however they end: a teardown that panics before
    /// or inside [closing](SessionCaptures::close) them only drops them, and every other session
    /// would otherwise be refused a capture on the device for the rest of the server's life.
    #[test]
    fn dropped_captures_hand_the_device_on() {
        let runner = TensorInterpreter::<Flex>::new(Default::default());
        let device = CaptureClaim::default();
        let stream = StreamId { value: 0 };

        let mut first = SessionCaptures::<Flex>::new(SessionId::new(), device.clone());
        first.prepare(&runner, stream).unwrap();
        drop(first);

        let mut second = SessionCaptures::<Flex>::new(SessionId::new(), device);
        second.prepare(&runner, stream).unwrap();
    }
}
