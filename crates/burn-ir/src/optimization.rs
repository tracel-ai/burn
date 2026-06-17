use crate::{ScalarIr, TensorId};
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Identifier for a cached, reusable group of operations (an "optimization").
///
/// A router backend that supports optimization caching (e.g. the remote backend) registers a
/// relative op-graph once under this id, then replays it by id with only the changing bindings —
/// see [`RouterClient::register_optimization`](crate::OptimizationId) usage in `burn-router`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OptimizationId(pub u64);

/// Per-invocation bindings used to specialize a cached optimization graph to concrete tensors.
///
/// The cached graph is in *relative* form (positional tensor ids, relative shape-dim ids, scalar
/// placeholders). The bindings carry only the graph's *boundary* — its inputs and surviving
/// outputs — plus the small relative→concrete shape-dim map and the scalar values. The replay
/// reconstructs the concrete shape of **every** tensor (including intermediates) from
/// [`shapes`](Self::shapes), and allocates fresh ids for intermediate tensors itself, so the
/// per-replay payload stays small regardless of how many ops the graph contains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationBindings {
    /// Boundary tensors only: `(relative id, concrete id)` for each graph input and surviving
    /// output. Intermediate tensors are *not* listed — the replaying backend allocates their ids.
    pub tensors: Vec<(TensorId, TensorId)>,
    /// Relative shape-dim id → concrete dim value, for every distinct dim in the graph. Lets the
    /// replay rebuild the concrete shape of every tensor, inputs/outputs and intermediates alike.
    pub shapes: Vec<(usize, usize)>,
    /// Concrete scalar values indexed by their placeholder id (the value carried in a relativized
    /// `ScalarIr::UInt(placeholder)`).
    pub scalars: Vec<ScalarIr>,
}
