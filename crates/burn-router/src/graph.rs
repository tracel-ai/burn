//! A cached relative op-graph and its replay against a [`TensorInterpreter`].
//!
//! A router server registers a recurring op sequence once as a [`Graph`] (in *relative* form:
//! positional tensor ids, relative shape dims, placeholder scalars/ranges), then replays it by id
//! with only the per-invocation [`GraphBindings`]. This is the server-side counterpart of the
//! client's cached optimization — it turns a recurring computation (e.g. a model block per step)
//! into one registration plus cheap replays.

use core::sync::atomic::{AtomicU64, Ordering};

use alloc::sync::Arc;
use alloc::vec::Vec;

use burn_backend::Slice;
use burn_ir::{BackendIr, GraphBindings, IrVisitorMut, OperationIr, ScalarIr, TensorId, TensorIr};
use hashbrown::HashMap;

use crate::TensorInterpreter;

/// Server-allocated ids for a replay's intermediate tensors carry this high bit so they can never
/// collide with client-allocated ids (whose monotonic counter never reaches `1 << 63`). The bit is
/// purely server-internal: intermediates are produced and freed within a single replay and are
/// never referenced by the client.
const INTERMEDIATE_ID_BIT: u64 = 1 << 63;
static INTERMEDIATE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn alloc_intermediate_id() -> TensorId {
    let value = INTERMEDIATE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    TensorId::new(value | INTERMEDIATE_ID_BIT)
}

/// A cached relative op-graph, registered once and [replayed](Graph::replay) by id with
/// per-invocation [`GraphBindings`].
///
/// Cheap to clone — the op list is shared behind an [`Arc`]. That lets a server keep its graph
/// cache behind a short-lived lock: clone the [`Graph`] handle out under the lock, then replay
/// after releasing it, so the lock never spans the backend dispatch.
#[derive(Clone, Debug)]
pub struct Graph {
    ops: Arc<Vec<OperationIr>>,
}

impl Graph {
    /// Wrap a relative op-graph so it can be replayed.
    pub fn new(ops: Vec<OperationIr>) -> Self {
        Self { ops: Arc::new(ops) }
    }

    /// Number of operations in the graph.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the graph has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Replay the graph against `interpreter`, rebinding the relative form to concrete tensors.
    ///
    /// The graph is in relative form: its tensor ids are positional, shape dims are relative ids,
    /// and scalars/ranges are placeholders. `bindings` arrive packaged the way the replay uses
    /// them — the boundary `tensors` map is moved straight into the working id table and grown with
    /// intermediate ids on demand, and `shapes` is a dense table indexed by relative dim id. For
    /// each op we:
    /// - resolve every tensor id to its boundary binding, or a freshly allocated intermediate id
    ///   (memoized so all references to one intermediate agree);
    /// - rewrite every tensor's shape dims in place via the shape table (so intermediates get
    ///   correct shapes too, without being sent);
    /// - substitute scalar placeholders and restore concrete slice ranges;
    ///
    /// then hand the rebound op to the unchanged [`TensorInterpreter::register_op`], reproducing the
    /// exact sequence of global ops the client would have streamed op-by-op.
    pub fn replay<B: BackendIr>(
        &self,
        interpreter: &mut TensorInterpreter<B>,
        bindings: GraphBindings,
    ) {
        let GraphBindings {
            tensors,
            shapes,
            scalars,
            ranges,
        } = bindings;
        // The boundary map *is* the working id table — seeded here, intermediates added on demand.
        let mut ids: HashMap<TensorId, TensorId> = tensors.into_iter().collect();
        for op in self.ops.iter() {
            let mut op = op.clone();
            let mut visitor = ReplayVisitor {
                ids: &mut ids,
                shapes: &shapes,
                scalars: &scalars,
                ranges: &ranges,
            };
            op.visit_mut(&mut visitor);
            interpreter.register_op(op);
        }
    }
}

/// Rebinds a relative op's tensors, scalars, and ranges to their concrete values during replay.
struct ReplayVisitor<'a> {
    /// The working id table; intermediates are allocated on demand and memoized here so all
    /// references to one intermediate agree. Persists across ops within a replay.
    ids: &'a mut HashMap<TensorId, TensorId>,
    shapes: &'a [usize],
    scalars: &'a [ScalarIr],
    ranges: &'a [Slice],
}

impl IrVisitorMut for ReplayVisitor<'_> {
    fn visit_tensor_mut(&mut self, tensor: &mut TensorIr) {
        tensor.id = *self
            .ids
            .entry(tensor.id)
            .or_insert_with(alloc_intermediate_id);
        for dim in tensor.shape.iter_mut() {
            *dim = self.shapes.get(*dim).copied().unwrap_or(*dim);
        }
    }

    fn visit_scalar_mut(&mut self, scalar: &mut ScalarIr) {
        if let ScalarIr::UInt(placeholder) = *scalar
            && (placeholder as usize) < self.scalars.len()
        {
            *scalar = self.scalars[placeholder as usize];
        }
    }

    fn visit_range_mut(&mut self, range: &mut Slice) {
        // Restore concrete slice bounds: relativization replaced each range with a placeholder
        // whose `start` is the binding id (see `OperationConverter::relative_range`).
        if let Some(concrete) = self.ranges.get(range.start as usize) {
            *range = *concrete;
        }
    }
}
