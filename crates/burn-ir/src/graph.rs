use crate::{ScalarIr, TensorId};
use alloc::vec::Vec;
use burn_backend::Slice;
use serde::{Deserialize, Serialize};

/// Identifier for a cached, reusable group of operations (a graph).
///
/// A router backend that supports graph caching (e.g. the remote backend) registers a relative
/// op-graph once under this id, then replays it by id with only the changing [bindings](GraphBindings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphId(pub u64);

/// Per-invocation bindings used to specialize a cached graph to concrete tensors.
///
/// The cached graph is in *relative* form (positional tensor ids, relative shape-dim ids, scalar
/// placeholders). The bindings carry only the graph's *boundary* — its inputs and surviving
/// outputs — plus the dense shape-dim table and the scalar values. The replay reconstructs the
/// concrete shape of **every** tensor (including intermediates) from [`shapes`](Self::shapes), and
/// allocates fresh ids for intermediate tensors itself, so the payload stays small regardless of
/// how many ops the graph contains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBindings {
    /// Boundary tensors only: `(relative id, concrete id)` for each graph input and surviving
    /// output. Intermediate tensors are *not* listed — the replaying backend allocates their ids.
    pub tensors: Vec<(TensorId, TensorId)>,
    /// Concrete dim value for each relative shape-dim id, indexed directly by the id. Relative dim
    /// ids are dense (`0..N`), so this is a plain table rather than a map, and each distinct dim
    /// value appears once however many tensors share it. Lets the replay rebuild every tensor's
    /// concrete shape — inputs, outputs and intermediates alike.
    pub shapes: Vec<usize>,
    /// Concrete scalar values indexed by their placeholder id (the value carried in a relativized
    /// `ScalarIr::UInt(placeholder)`).
    pub scalars: Vec<ScalarIr>,
    /// Concrete slice ranges indexed by their placeholder id (the value carried in a relativized
    /// range's `start` field). The relative graph keeps every `Slice` range as a positional
    /// placeholder — its actual bounds are discarded by relativization (they vary per invocation,
    /// like scalars) — so the replay restores each from here.
    pub ranges: Vec<Slice>,
}
