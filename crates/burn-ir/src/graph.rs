use crate::{OperationIr, ScalarIr, TensorId, TensorStatus};
use alloc::vec::Vec;
use burn_backend::Slice;
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};

/// An ordered operation graph with an explicitly classified tensor boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphIr {
    /// Operations in execution order.
    pub operations: Vec<OperationIr>,
    /// Tensors read by the graph but not produced by a compute operation.
    pub inputs: Vec<TensorId>,
    /// Compute-produced tensors which survive the graph.
    pub outputs: Vec<TensorId>,
}

impl GraphIr {
    /// Normalize an operation sequence and classify its tensor boundary in first-use order.
    ///
    /// Inputs are tensors that aren't produced by a compute operation: external data and results
    /// from an earlier graph, including outputs of [`OperationIr::Init`]. Initializer handles are
    /// registered out of band, so they remain graph inputs even though an initialization operation
    /// produces them.
    ///
    /// Outputs are tensors produced by compute operations that survive the graph. Tensors consumed
    /// in place (`ReadWrite`) or explicitly dropped within the graph are excluded.
    ///
    /// Intermediate tensors that are both produced and consumed within the graph appear in neither
    /// boundary. Input and output identifiers retain their first-use and production order,
    /// respectively, making construction deterministic.
    pub fn new(operations: Vec<OperationIr>) -> Self {
        let mut referenced = Vec::new();
        let mut referenced_set = HashSet::new();
        let mut produced = Vec::new();
        let mut produced_set = HashSet::new();
        let mut consumed = HashSet::new();

        for operation in &operations {
            if let OperationIr::Drop(tensor) = operation {
                consumed.insert(tensor.id);
            }
            if !matches!(operation, OperationIr::Init(_)) {
                for tensor in operation.outputs() {
                    if produced_set.insert(tensor.id) {
                        produced.push(tensor.id);
                    }
                }
            }
            for tensor in operation.nodes() {
                if referenced_set.insert(tensor.id) {
                    referenced.push(tensor.id);
                }
                if tensor.status == TensorStatus::ReadWrite {
                    consumed.insert(tensor.id);
                }
            }
        }

        let inputs = referenced
            .into_iter()
            .filter(|id| !produced_set.contains(id))
            .collect();
        let outputs = produced
            .into_iter()
            .filter(|id| !consumed.contains(id))
            .collect();
        Self {
            operations,
            inputs,
            outputs,
        }
    }

    /// Number of operations in the graph.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Whether the graph contains no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

impl From<Vec<OperationIr>> for GraphIr {
    fn from(operations: Vec<OperationIr>) -> Self {
        Self::new(operations)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CustomOpIr, TensorIr};
    use burn_backend::{DType, Shape};

    fn tensor(id: u64) -> TensorIr {
        TensorIr::uninit(TensorId::new(id), Shape::new([1]), DType::F32)
    }

    #[test]
    fn graph_boundary_is_in_first_use_order() {
        let first = OperationIr::Custom(CustomOpIr::new(
            "first",
            &[tensor(9), tensor(3)],
            &[tensor(5)],
        ));
        let second = OperationIr::Custom(CustomOpIr::new("second", &[tensor(5)], &[tensor(7)]));
        let graph = GraphIr::new(vec![first, second, OperationIr::Drop(tensor(5))]);

        assert_eq!(graph.inputs, vec![TensorId::new(9), TensorId::new(3)]);
        assert_eq!(graph.outputs, vec![TensorId::new(7)]);
    }

    #[test]
    fn graph_normalization_is_deterministic() {
        let operations = vec![OperationIr::Custom(CustomOpIr::new(
            "op",
            &[tensor(4), tensor(2), tensor(4)],
            &[tensor(8)],
        ))];
        assert_eq!(GraphIr::new(operations.clone()), GraphIr::new(operations));
    }
}
