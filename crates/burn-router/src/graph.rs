//! A cached relative op-graph and its replay against a [`TensorInterpreter`].
//!
//! A router server registers a recurring op sequence once as a [`Graph`] (in *relative* form:
//! positional tensor ids, relative shape dims, placeholder scalars/ranges), then replays it by id
//! with only the per-invocation [`GraphBindings`]. This is the server-side counterpart of the
//! client's cached optimization — it turns a recurring computation (e.g. a model block per step)
//! into one registration plus cheap replays.

use alloc::sync::Arc;
use alloc::vec::Vec;

use burn_backend::Slice;
use burn_ir::{
    BackendIr, GraphBindings, GraphIr, IrVisitorMut, OperationIr, ScalarIr, TensorId, TensorIr,
};
use hashbrown::HashMap;
use portable_atomic::{AtomicU64, Ordering};

use crate::TensorInterpreter;

/// Router-allocated ids for a replay's intermediate tensors carry this high bit so they can never
/// collide with client-allocated ids (whose monotonic counter never reaches `1 << 63`). Executing
/// servers keep these ids internal, while non-executing consumers such as graph capture may expose
/// them as opaque ids in the bound concrete graph.
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
    graph: Arc<GraphIr>,
}

impl Graph {
    /// Wrap a relative op-graph so it can be replayed.
    pub fn new(ops: Vec<OperationIr>) -> Self {
        Self {
            graph: Arc::new(GraphIr::new(ops)),
        }
    }

    /// Number of operations in the graph.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Whether the graph has no operations.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Return the relative operation sequence.
    pub fn operations(&self) -> &[OperationIr] {
        &self.graph.operations
    }

    /// Return the relative input boundary.
    pub fn inputs(&self) -> &[TensorId] {
        &self.graph.inputs
    }

    /// Return the relative output boundary.
    pub fn outputs(&self) -> &[TensorId] {
        &self.graph.outputs
    }

    /// Bind the relative graph to one invocation and return a concrete graph.
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
    /// The returned graph can be inspected by a non-executing consumer such as graph capture.
    /// Binding itself performs no tensor computation.
    pub fn bind(&self, bindings: GraphBindings) -> GraphIr {
        let mut operations = Vec::with_capacity(self.graph.operations.len());
        self.for_each_bound_operation(bindings, |operation| operations.push(operation));
        GraphIr::new(operations)
    }

    /// Visit each operation after binding it to a concrete invocation.
    ///
    /// Keeping the shared traversal callback-based lets executing router servers stream bound
    /// operations directly into their interpreter without first allocating a second operation
    /// vector. Non-executing consumers use [`bind`](Self::bind) to collect
    /// the same traversal.
    fn for_each_bound_operation(
        &self,
        bindings: GraphBindings,
        mut visit: impl FnMut(OperationIr),
    ) {
        let GraphBindings {
            tensors,
            shapes,
            scalars,
            ranges,
        } = bindings;
        // The boundary map *is* the working id table — seeded here, intermediates added on demand.
        let mut ids: HashMap<TensorId, TensorId> = tensors.into_iter().collect();
        for op in self.graph.operations.iter() {
            let mut op = op.clone();
            let mut visitor = BindingVisitor {
                ids: &mut ids,
                shapes: &shapes,
                scalars: &scalars,
                ranges: &ranges,
            };
            op.visit_mut(&mut visitor);
            visit(op);
        }
    }

    /// Replay the graph against `interpreter` after binding it to concrete tensors.
    ///
    /// This is the executing counterpart of [`bind`](Self::bind): it hands
    /// each bound operation to [`TensorInterpreter::register_op`] in its original order.
    pub fn replay<B: BackendIr>(
        &self,
        interpreter: &mut TensorInterpreter<B>,
        bindings: GraphBindings,
    ) {
        self.for_each_bound_operation(bindings, |operation| interpreter.register_op(operation));
    }
}

/// Binds a relative op's tensors, scalars, and ranges to their invocation values.
struct BindingVisitor<'a> {
    /// The working id table; intermediates are allocated on demand and memoized here so all
    /// references to one intermediate agree. Persists across ops within a replay.
    ids: &'a mut HashMap<TensorId, TensorId>,
    shapes: &'a [usize],
    scalars: &'a [ScalarIr],
    ranges: &'a [Slice],
}

impl IrVisitorMut for BindingVisitor<'_> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, Shape, Slice};
    use burn_ir::{
        BaseOperationIr, CustomOpIr, NumericOperationIr, PadModeIr, PadOpIr, ScalarIr, SliceOpIr,
        TensorStatus,
    };

    fn tensor(id: u64, shape: [usize; 2]) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::new(shape),
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        }
    }

    #[test]
    fn bind_resolves_tensors_shapes_scalars_and_ranges() {
        let relative_input = tensor(0, [0, 1]);
        let relative_intermediate = tensor(1, [0, 1]);
        let relative_output = tensor(2, [0, 1]);
        let graph = Graph::new(vec![
            OperationIr::BaseFloat(BaseOperationIr::Slice(SliceOpIr {
                tensor: relative_input.clone(),
                ranges: vec![Slice::new(0, None, 1)],
                out: relative_intermediate.clone(),
            })),
            OperationIr::Custom(CustomOpIr::with_scalars(
                "bound",
                core::slice::from_ref(&relative_intermediate),
                core::slice::from_ref(&relative_output),
                vec![ScalarIr::UInt(0)],
            )),
        ]);
        let concrete_input = TensorId::new(100);
        let concrete_output = TensorId::new(102);
        let concrete_range = Slice::new(2, Some(8), 2);

        let bound = graph.bind(GraphBindings {
            tensors: vec![
                (relative_input.id, concrete_input),
                (relative_output.id, concrete_output),
            ],
            shapes: vec![4, 8],
            scalars: vec![ScalarIr::Int(12)],
            ranges: vec![concrete_range],
        });
        let operations = &bound.operations;

        let OperationIr::BaseFloat(BaseOperationIr::Slice(slice)) = &operations[0] else {
            panic!("expected bound slice")
        };
        let OperationIr::Custom(custom) = &operations[1] else {
            panic!("expected bound custom operation")
        };
        assert_eq!(slice.tensor.id, concrete_input);
        assert_eq!(slice.tensor.shape, Shape::new([4, 8]));
        assert_eq!(slice.ranges, [concrete_range]);
        assert_eq!(slice.out.id, custom.inputs[0].id);
        assert_ne!(slice.out.id, relative_intermediate.id);
        assert_eq!(custom.outputs[0].id, concrete_output);
        assert_eq!(custom.scalars, [ScalarIr::Int(12)]);
        assert_eq!(bound.inputs, [concrete_input]);
        assert!(bound.outputs.contains(&concrete_output));
    }

    #[test]
    fn each_binding_allocates_fresh_intermediate_ids() {
        let relative_input = tensor(0, [0, 0]);
        let relative_intermediate = tensor(1, [0, 0]);
        let relative_output = tensor(2, [0, 0]);
        let graph = Graph::new(vec![
            OperationIr::Custom(CustomOpIr::new(
                "first",
                core::slice::from_ref(&relative_input),
                core::slice::from_ref(&relative_intermediate),
            )),
            OperationIr::Custom(CustomOpIr::new(
                "second",
                &[relative_intermediate],
                core::slice::from_ref(&relative_output),
            )),
        ]);
        let bind = |input, output| GraphBindings {
            tensors: vec![
                (relative_input.id, TensorId::new(input)),
                (relative_output.id, TensorId::new(output)),
            ],
            shapes: vec![2],
            scalars: vec![],
            ranges: vec![],
        };

        let first = graph.bind(bind(10, 12));
        let second = graph.bind(bind(20, 22));
        let OperationIr::Custom(first_head) = &first.operations[0] else {
            unreachable!()
        };
        let OperationIr::Custom(first_tail) = &first.operations[1] else {
            unreachable!()
        };
        let OperationIr::Custom(second_head) = &second.operations[0] else {
            unreachable!()
        };
        assert_eq!(first_head.outputs[0].id, first_tail.inputs[0].id);
        assert_ne!(first_head.outputs[0].id, second_head.outputs[0].id);
    }

    #[test]
    fn bind_resolves_pad_constant_scalar() {
        let relative_input = tensor(0, [0, 1]);
        let relative_output = tensor(1, [2, 3]);
        let graph = Graph::new(vec![OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Pad(PadOpIr {
                input: relative_input.clone(),
                out: relative_output.clone(),
                padding: vec![(1, 0), (0, 2)],
                mode: PadModeIr::Constant(ScalarIr::UInt(0)),
            }),
        )]);

        let bound = graph.bind(GraphBindings {
            tensors: vec![
                (relative_input.id, TensorId::new(100)),
                (relative_output.id, TensorId::new(101)),
            ],
            shapes: vec![2, 3, 3, 5],
            scalars: vec![ScalarIr::Float(6.5)],
            ranges: vec![],
        });

        let OperationIr::NumericFloat(_, NumericOperationIr::Pad(desc)) = &bound.operations[0]
        else {
            panic!("expected bound pad operation");
        };
        assert_eq!(desc.input.id, TensorId::new(100));
        assert_eq!(desc.out.id, TensorId::new(101));
        assert_eq!(desc.input.shape, Shape::new([2, 3]));
        assert_eq!(desc.out.shape, Shape::new([3, 5]));
        assert_eq!(desc.mode, PadModeIr::Constant(ScalarIr::Float(6.5)));
    }
}
