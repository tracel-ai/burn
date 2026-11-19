use std::{collections::HashMap, sync::Mutex, thread};

use burn_backend::{BackendGraph, ExecutionError};
use burn_ir::{BackendIr, CapturedGraphId};
use burn_router::TensorInterpreter;
use burn_std::id::{IdGenerator, StreamId};

/// The graphs a sharded compute device captured, one per member under each id.
///
/// A capture records every member at once: one window open on each member's stream, so a
/// collective recorded on one member is recorded on all of them, in the order the interpreter
/// issued it. A replay replays every member's graph, since a member left out would leave its
/// peers waiting inside that collective.
pub struct MemberGraphs<B: BackendIr> {
    graphs: HashMap<CapturedGraphId, Vec<BackendGraph<B>>>,
}

impl<B: BackendIr> Default for MemberGraphs<B> {
    fn default() -> Self {
        Self {
            graphs: HashMap::new(),
        }
    }
}

impl<B: BackendIr> MemberGraphs<B> {
    /// Prepares every member for a capture and starts planning, and returns the first
    /// member's error.
    ///
    /// A member that refuses ends the captures its predecessors prepared, since a prepared member
    /// discards its launches.
    pub fn prepare(&self, members: &[TensorInterpreter<B>]) -> Result<(), ExecutionError> {
        for (index, member) in members.iter().enumerate() {
            if let Err(error) = member.graph_prepare() {
                stop_every(&members[..index]);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Ends one planning run on every member, and answers whether every member's settled. Every
    /// member's run is ended whichever answers first, since a run left open spans the next one.
    ///
    /// A member that fails to end its run ends its own capture (see
    /// [`Backend::graph_end_run`](burn_backend::Backend::graph_end_run)), so the others' are
    /// ended too, and the work runs on every member again.
    pub fn end_run(&self, members: &[TensorInterpreter<B>]) -> Result<bool, ExecutionError> {
        let settled = members.iter().fold(Ok(true), |settled, member| {
            let answer = member.graph_end_run();
            Ok(settled? & answer?)
        });
        if settled.is_err() {
            stop_every(members);
        }
        settled
    }

    /// Opens a window on every member, or none: a member that refuses closes the windows its
    /// predecessors opened, discarding what they hold.
    pub fn start(&self, members: &[TensorInterpreter<B>]) -> Result<(), ExecutionError> {
        for (index, member) in members.iter().enumerate() {
            // A member that refuses ends its own capture; the others, recording or still
            // prepared, are ended with it, since a prepared member discards its launches.
            if let Err(error) = member.graph_start_capture() {
                stop_every(&members[..index]);
                stop_every(&members[index + 1..]);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Closes every member's window, and keeps their graphs under a new id when all of them
    /// recorded one. Every window is closed whichever fails first, since a stream left capturing
    /// records whatever follows.
    pub fn stop(
        &mut self,
        members: &[TensorInterpreter<B>],
    ) -> Result<CapturedGraphId, ExecutionError> {
        let stopped: Vec<_> = members
            .iter()
            .map(TensorInterpreter::graph_stop_capture)
            .collect();
        let graphs = stopped.into_iter().collect::<Result<Vec<_>, _>>()?;
        let id = CapturedGraphId(IdGenerator::generate());
        self.graphs.insert(id, graphs);
        Ok(id)
    }

    /// Replays every member's graph of `id` at once, and returns the error of a member that
    /// failed. A member that fails does not stop the others replaying, whose peers would otherwise
    /// wait inside a collective for it.
    ///
    /// # Safety
    ///
    /// The contract of [`Backend::graph_replay`](burn_backend::Backend::graph_replay), for every
    /// member's tensors the graph read or wrote.
    pub unsafe fn replay(
        &self,
        members: &[TensorInterpreter<B>],
        id: CapturedGraphId,
    ) -> Result<(), ExecutionError> {
        let graphs = self.graphs.get(&id).ok_or_else(|| {
            ExecutionError::with_context(format!("No graph was captured under {id:?}"))
        })?;
        let first_error = Mutex::new(None);
        let stream = StreamId::current();
        // Launched in turn, members idle at their first collective until the last launch returns,
        // and a launch may block until its peers have launched.
        thread::scope(|scope| {
            for (member, graph) in members.iter().zip(graphs) {
                let first_error = &first_error;
                scope.spawn(move || {
                    // Safety: forwarded verbatim from this method's own contract, on its stream.
                    let replayed = stream.executes(|| unsafe { member.graph_replay(graph) });
                    if let Err(error) = replayed {
                        first_error.lock().unwrap().get_or_insert(error);
                    }
                });
            }
        });
        first_error.into_inner().unwrap().map_or(Ok(()), Err)
    }

    /// Frees every member's graph of `id`.
    pub fn release(&mut self, id: CapturedGraphId) {
        self.graphs.remove(&id);
    }
}

/// End the capture of every one of `members`, recording or prepared, whatever it holds: a member
/// left prepared discards its launches.
fn stop_every<B: BackendIr>(members: &[TensorInterpreter<B>]) {
    for member in members {
        let _ = member.graph_stop_capture();
    }
}
