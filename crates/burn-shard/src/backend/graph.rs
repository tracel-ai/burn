use std::collections::HashMap;

use burn_backend::{BackendGraph, ExecutionError};
use burn_ir::{BackendIr, CapturedGraphId};
use burn_router::TensorInterpreter;
use burn_std::id::IdGenerator;

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
    /// Prepares every member for a capture, and returns the first member's error.
    pub fn prepare(&self, members: &[TensorInterpreter<B>]) -> Result<(), ExecutionError> {
        members
            .iter()
            .try_for_each(TensorInterpreter::graph_prepare)
    }

    /// Opens a window on every member, or none: a member that refuses closes the windows its
    /// predecessors opened, discarding what they hold.
    pub fn start(&self, members: &[TensorInterpreter<B>]) -> Result<(), ExecutionError> {
        for (index, member) in members.iter().enumerate() {
            if let Err(error) = member.graph_start_capture() {
                members[..index].iter().for_each(|opened| {
                    let _ = opened.graph_stop_capture();
                });
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

    /// Replays every member's graph of `id`, in member order, and returns the first member's
    /// error. A member that fails does not stop the others replaying, whose peers would otherwise
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
        let mut first_error = None;
        for (member, graph) in members.iter().zip(graphs) {
            // Safety: forwarded verbatim from this method's own contract.
            if let Err(error) = unsafe { member.graph_replay(graph) } {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Frees every member's graph of `id`.
    pub fn release(&mut self, id: CapturedGraphId) {
        self.graphs.remove(&id);
    }
}
