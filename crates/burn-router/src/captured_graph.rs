use crate::RouterClient;
use alloc::sync::Arc;
use burn_backend::ExecutionError;
use burn_ir::CapturedGraphId;

/// The [graph primitive](burn_backend::BackendTypes::GraphPrimitive) of the
/// [router backend](crate::BackendRouter): a graph captured on the backend behind a client's
/// interpreter, named by the id the interpreter keeps it under.
///
/// Clones share the graph. Dropping the last one [releases](RouterClient::graph_release) it on
/// the interpreter, after every operation the client registered before.
pub struct RouterGraph<C: RouterClient> {
    captured: Arc<CapturedGraph<C>>,
}

/// The id of a captured graph and the client whose interpreter holds it, released on drop.
struct CapturedGraph<C: RouterClient> {
    id: CapturedGraphId,
    client: C,
}

impl<C: RouterClient> RouterGraph<C> {
    /// The graph `client`'s interpreter keeps under `id`, which this value now owns.
    pub fn new(id: CapturedGraphId, client: C) -> Self {
        Self {
            captured: Arc::new(CapturedGraph { id, client }),
        }
    }

    /// Replay the graph on the interpreter that captured it, after every operation registered
    /// so far on the calling stream.
    ///
    /// # Safety
    ///
    /// The contract of [`Backend::graph_replay`](burn_backend::Backend::graph_replay).
    pub unsafe fn replay(&self) -> Result<(), ExecutionError> {
        // Safety: forwarded verbatim from this method's own contract.
        unsafe { self.captured.client.graph_replay(self.captured.id) }
    }
}

impl<C: RouterClient> Drop for CapturedGraph<C> {
    fn drop(&mut self) {
        self.client.graph_release(self.id);
    }
}

impl<C: RouterClient> Clone for RouterGraph<C> {
    fn clone(&self) -> Self {
        Self {
            captured: self.captured.clone(),
        }
    }
}

impl<C: RouterClient> core::fmt::Debug for RouterGraph<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("RouterGraph")
            .field(&self.captured.id)
            .finish()
    }
}
