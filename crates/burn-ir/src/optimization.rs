use crate::{ScalarIr, TensorId, TensorIr};
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
/// The cached graph is in *relative* form (positional tensor ids, relative shape dims, scalar
/// placeholders). On each replay every relative tensor id is mapped to a concrete tensor id +
/// global shape, and every scalar placeholder to a concrete value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationBindings {
    /// For each relative tensor id (the `id` carried by tensors in the cached graph), the concrete
    /// tensor to substitute. Only the `id` and `shape` are used on replay — the `status`/`dtype`
    /// come from the cached op.
    pub tensors: Vec<(TensorId, TensorIr)>,
    /// Concrete scalar values indexed by their placeholder id (the value carried in a relativized
    /// `ScalarIr::UInt(placeholder)`).
    pub scalars: Vec<ScalarIr>,
}
