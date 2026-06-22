//! Generic client-side graph caching for any [router backend](crate::BackendRouter).
//!
//! Wrapping a router backend as [`Fusion`](burn_fusion::Fusion) turns recurring groups of tensor
//! operations into reusable, client-cached graphs. A single greedy [`RouterFuser`] accumulates
//! every operation and is drained only at a sync point, so one graph covers each connected block
//! between syncs. On execution the group is registered once on the backend (via the
//! [`RouterClient`]) and thereafter invoked by id with only the changing bindings — for the remote
//! backend this means a recurring computation (e.g. a model block) crosses the network once
//! instead of every step.
//!
//! `burn-fusion` names its hook `Optimization`; here that hook *is* a cached graph execution
//! ([`RouterGraphExecution`]), to distinguish it from a compute backend's kernel fusion.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use burn_backend::DType;
use burn_backend::ops::FloatTensorOps;
use burn_fusion::stream::{Context, OrderedExecution};
use burn_fusion::{
    FuserProperties, FuserStatus, FusionBackend, FusionRuntime, NumOperations, OperationFuser,
    Optimization,
};
use burn_ir::{
    BackendIr, GraphBindings, GraphId, OperationIr, ScalarIr, TensorHandle, TensorId, TensorStatus,
};
use serde::{Deserialize, Serialize};

use burn_std::config::config;

use crate::{BackendRouter, RouterChannel, RouterClient, RouterTensor, get_client};

static GRAPH_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_graph_id() -> GraphId {
    GraphId(GRAPH_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

// The router backend already implements `Backend`; these two impls add the `BackendIr` +
// `FusionBackend` glue so it can be wrapped as `Fusion<BackendRouter<R>>`. All four tensor
// primitives are `RouterTensor`, so the handle conversions are the identity (mirroring the
// `impl BackendIr for Fusion<B>` in burn-fusion).
impl<R: RouterChannel> BackendIr for BackendRouter<R> {
    type Handle = RouterTensor<R::Client>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> burn_backend::tensor::FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> burn_backend::tensor::IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> burn_backend::tensor::BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(
        handle: TensorHandle<Self::Handle>,
    ) -> burn_backend::tensor::QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: burn_backend::tensor::FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: burn_backend::tensor::IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: burn_backend::tensor::BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(
        tensor: burn_backend::tensor::QuantizedTensor<Self>,
    ) -> Self::Handle {
        tensor
    }
}

impl<R: RouterChannel> FusionBackend for BackendRouter<R> {
    type FusionRuntime = RouterFusionRuntime<R>;
    type FullPrecisionBackend = Self;

    fn cast_float(tensor: burn_backend::tensor::FloatTensor<Self>, dtype: DType) -> Self::Handle {
        Self::float_cast(tensor, dtype.into())
    }
}

/// The [fusion runtime](FusionRuntime) for a [router backend](BackendRouter).
///
/// Its [handle](FusionRuntime::FusionHandle) is a [`RouterTensor`] — a lightweight reference to a
/// backend-resident tensor plus the client to reach it — and its "optimization" is a backend-cached
/// op-graph (see [`RouterGraphExecution`]).
pub struct RouterFusionRuntime<R: RouterChannel> {
    _p: PhantomData<R>,
}

impl<R: RouterChannel> core::fmt::Debug for RouterFusionRuntime<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RouterFusionRuntime")
    }
}

impl<R: RouterChannel> FusionRuntime for RouterFusionRuntime<R> {
    type OptimizationState = RouterGraphExecutionState;
    type Optimization = RouterGraphExecution<R>;
    type FusionHandle = RouterTensor<R::Client>;
    type FusionDevice = R::Device;

    fn fusers(device: R::Device) -> Vec<Box<dyn OperationFuser<Self::Optimization>>> {
        vec![Box::new(RouterFuser::<R>::new(device))]
    }

    fn alias_handle(handle: &RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        // The router handle is a thin id into a server-side tensor, so a bare `clone()` would keep
        // the same server id — every cross-stream alias would collapse onto one server handle, and
        // the first stream to consume it (`ReadWrite`) would free it for the others. Instead mint a
        // fresh id and have the server register it as an alias of the same buffer (a refcounted
        // clone), so each stream's view frees independently. Mirrors local backends, where the
        // default `clone()` already yields an independent `HandleContainer` entry over a shared
        // `Arc` buffer.
        let id = handle.client.create_empty_handle();
        handle.client.register_alias(id, handle.id);
        RouterTensor::new(
            id,
            handle.shape.clone(),
            handle.dtype,
            handle.client.clone(),
        )
    }
}

/// A greedy operation fuser that records every operation and never closes itself.
///
/// Because [`status`](OperationFuser::status) always reports [`FuserStatus::Open`], the fusion
/// engine keeps deferring in lazy mode and only drains the queue at a sync point (a read, `sync`,
/// or `flush`). At that point [`finish`](OperationFuser::finish) yields a single
/// [`RouterGraphExecution`] covering the whole accumulated (connected) block.
pub struct RouterFuser<R: RouterChannel> {
    device: R::Device,
    ops: Vec<OperationIr>,
    score: u64,
    score_max: u64,
    num_since_max_unchanged: usize,
    /// Close the graph once it reaches this many ops, if set (`FusionConfig::max_graph_size`).
    max_graph_size: Option<usize>,
    /// Close the graph once the score hasn't reached a new max for this many consecutive ops
    /// (`FusionConfig::growth_patience`).
    growth_patience: usize,
}

impl<R: RouterChannel> RouterFuser<R> {
    fn new(device: R::Device) -> Self {
        let cfg = config();
        let fusion = cfg.fusion();
        Self {
            device,
            ops: Vec::new(),
            score: 0,
            score_max: 0,
            num_since_max_unchanged: 0,
            max_graph_size: fusion.max_graph_size,
            growth_patience: fusion.growth_patience,
        }
    }

    /// Value-based fusion score for the currently accumulated ops.
    ///
    /// Benefit: estimated % of serialized bytes caching saves per replay (the relative graph vs the
    /// per-replay bindings — see [`estimate_saved_pct`]), weighted by `FACTOR_SAVED`. Cost: a
    /// penalty that grows with op count past `FREE_OPS`, so the score *peaks* at a "right-sized"
    /// graph and then decays once the per-op overhead outweighs the savings.
    ///
    /// Floored at 1 for any non-empty graph: a score of 0 makes `find_best_optimization_index`
    /// treat the graph as "don't fuse" (streaming every op unfused → the cache never replays). The
    /// `+1` doesn't move the argmax, so it doesn't change which size wins.
    fn score(&self) -> u64 {
        const FACTOR_SAVED: u64 = 100; // weight per % point saved → benefit in 0..=10_000
        const FREE_OPS: usize = 64; // ops below this are not penalized
        const PENALTY_PER_OP: u64 = 0; // penalty per op beyond FREE_OPS

        let benefit = estimate_saved_pct(&self.ops) * FACTOR_SAVED;
        let penalty = (self.ops.len().saturating_sub(FREE_OPS) as u64) * PENALTY_PER_OP;
        benefit.saturating_sub(penalty) + 1
    }
}

impl<R: RouterChannel> Clone for RouterFuser<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            ops: self.ops.clone(),
            score: self.score,
            score_max: self.score_max,
            num_since_max_unchanged: self.num_since_max_unchanged,
            max_graph_size: self.max_graph_size,
            growth_patience: self.growth_patience,
        }
    }
}

impl<R: RouterChannel> OperationFuser<RouterGraphExecution<R>> for RouterFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        self.ops.push(operation.clone());

        self.score = self.score();
        if self.score > self.score_max {
            self.score_max = self.score;
            self.num_since_max_unchanged = 0;
        } else {
            self.num_since_max_unchanged += 1;
        }
    }

    fn finish(&mut self) -> RouterGraphExecution<R> {
        let ops = core::mem::take(&mut self.ops);
        RouterGraphExecution::new(ops, self.device.clone())
    }

    fn reset(&mut self) {
        self.ops.clear();
    }

    fn status(&self) -> FuserStatus {
        let over_max = self.max_graph_size.is_some_and(|max| self.len() > max);
        if self.num_since_max_unchanged >= self.growth_patience || over_max {
            FuserStatus::Closed
        } else {
            FuserStatus::Open
        }
    }

    fn properties(&self) -> FuserProperties {
        FuserProperties {
            score: self.score,
            ready: self.ops.len() > 1,
        }
    }

    fn len(&self) -> usize {
        self.ops.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<RouterGraphExecution<R>>> {
        Box::new(self.clone())
    }
}

/// A reusable group of operations, registered once on the backend and thereafter invoked by id.
///
/// The recorded [`graph`](Self::graph) is in *relative* form (positional tensor ids, relative
/// shape-dim ids, scalar placeholders), which is invariant across invocations — that is what lets
/// the backend cache it and the client reuse it. On each [`execute`](Optimization::execute) only
/// the concrete bindings are computed from the [`Context`] and sent; the (large) graph itself
/// travels only on the first invocation.
pub struct RouterGraphExecution<R: RouterChannel> {
    graph: Vec<OperationIr>,
    device: R::Device,
    /// Relative ids of the graph's boundary, precomputed once from the (static) graph so each
    /// replay is O(boundary) instead of re-scanning every tensor. Inputs are tensors not produced
    /// by a compute op (external data / prior results, incl. `Init`/`from_data` outputs); outputs
    /// are compute-produced tensors that survive (neither consumed in place nor dropped here).
    input_ids: Vec<TensorId>,
    output_ids: Vec<TensorId>,
    /// Backend-side id, assigned and registered on the first execution and reused afterwards.
    graph_id: Option<GraphId>,
    _p: PhantomData<R>,
}

/// Serializable state for a [`RouterGraphExecution`].
///
/// The backend id is intentionally not serialized — it is per-connection state, so a deserialized
/// graph re-registers itself on first use.
#[derive(Serialize, Deserialize)]
pub struct RouterGraphExecutionState {
    graph: Vec<OperationIr>,
}

impl<R: RouterChannel> RouterGraphExecution<R> {
    fn new(graph: Vec<OperationIr>, device: R::Device) -> Self {
        let (input_ids, output_ids) = classify_boundary(&graph);
        Self {
            graph,
            device,
            input_ids,
            output_ids,
            graph_id: None,
            _p: PhantomData,
        }
    }
}

/// Precompute the graph's boundary as `(input relative ids, surviving-output relative ids)`.
///
/// - **Inputs** are tensors that aren't produced by a *compute* op: external data and prior-block
///   results read by the graph, plus `Init`/`from_data` outputs (whose handle is registered
///   out-of-band, so they're really inputs even though an `Init` op "produces" them).
/// - **Outputs** are compute-produced tensors that survive — neither consumed in place
///   (`ReadWrite` anywhere) nor dropped within the graph. This mirrors the fusion engine's own
///   `drain_queue` freeing logic, keeping client-side handle state consistent with the unfused path.
///
/// Intermediate tensors (compute-produced and consumed/dropped here) are in neither list — the
/// replay owns their ids — so a replay only ever touches the boundary.
fn classify_boundary(graph: &[OperationIr]) -> (Vec<TensorId>, Vec<TensorId>) {
    let mut referenced: HashSet<TensorId> = HashSet::new();
    let mut compute_produced: HashSet<TensorId> = HashSet::new();
    let mut consumed: HashSet<TensorId> = HashSet::new();
    for op in graph {
        if let OperationIr::Drop(tensor) = op {
            consumed.insert(tensor.id);
        }
        if !matches!(op, OperationIr::Init(_)) {
            for tensor in op.outputs() {
                compute_produced.insert(tensor.id);
            }
        }
        for tensor in op.nodes() {
            referenced.insert(tensor.id);
            if tensor.status == TensorStatus::ReadWrite {
                consumed.insert(tensor.id);
            }
        }
    }

    let inputs = referenced
        .iter()
        .filter(|id| !compute_produced.contains(id))
        .copied()
        .collect();
    let outputs = compute_produced
        .iter()
        .filter(|id| !consumed.contains(id))
        .copied()
        .collect();
    (inputs, outputs)
}

/// Estimate the % of serialized bytes that caching this op-graph saves per replay.
///
/// Dependency-free structural estimate (reuses [`classify_boundary`]): the baseline is the whole
/// relative graph's bytes (op overhead + each tensor's id/dtype/status + its dims); the per-replay
/// bindings are just the boundary `(relative id, concrete id)` pairs plus the distinct dim table.
/// Scalars/ranges travel in both and cancel in the ratio. Returns 0..=100.
fn estimate_saved_pct(ops: &[OperationIr]) -> u64 {
    if ops.is_empty() {
        return 0;
    }
    const TENSOR: u64 = 10; // id (≈8) + dtype + status
    const PER_DIM: u64 = 8;
    const OP_OVERHEAD: u64 = 8; // variant tag + small bookkeeping
    const PAIR: u64 = 16; // a (relative id, concrete id) binding entry

    let mut baseline = 0u64;
    let mut dims: HashSet<usize> = HashSet::new();
    for op in ops {
        baseline += OP_OVERHEAD;
        for tensor in op.nodes().into_iter().chain(op.outputs()) {
            baseline += TENSOR;
            for dim in tensor.shape.iter() {
                baseline += PER_DIM;
                dims.insert(*dim);
            }
        }
    }

    let (inputs, outputs) = classify_boundary(ops);
    let bindings = (inputs.len() + outputs.len()) as u64 * PAIR + dims.len() as u64 * PER_DIM;

    let saved = baseline.saturating_sub(bindings);
    (saved * 100 / baseline).min(100)
}

impl<R: RouterChannel> core::fmt::Debug for RouterGraphExecution<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RouterGraphExecution")
            .field("len", &self.graph.len())
            .finish()
    }
}

impl<R: RouterChannel> NumOperations for RouterGraphExecution<R> {
    fn len(&self) -> usize {
        self.graph.len()
    }

    fn name(&self) -> &'static str {
        "RouterGraphExecution"
    }
}

impl<R: RouterChannel> Optimization<RouterFusionRuntime<R>> for RouterGraphExecution<R> {
    fn execute(
        &mut self,
        context: &mut Context<RouterTensor<R::Client>>,
        _execution: &OrderedExecution<RouterFusionRuntime<R>>,
    ) {
        let client = get_client::<R>(&self.device);

        // Walk only the precomputed boundary — never every tensor — so a replay is O(boundary).
        // Inputs reuse their resident concrete id; surviving outputs get a fresh id and a handle.
        // Intermediates are never touched here: the replay owns their ids and derives their shapes
        // from the shape table below.
        let mut tensors: Vec<(TensorId, TensorId)> =
            Vec::with_capacity(self.input_ids.len() + self.output_ids.len());

        for &input_id in &self.input_ids {
            let global_id = match context.tensors.get(&input_id) {
                Some(global) => global.id,
                None => continue,
            };
            if let Some(handle) = context.handles.get_handle_ref(&global_id) {
                tensors.push((input_id, handle.id()));
            }
        }

        for &output_id in &self.output_ids {
            // Extract owned metadata first so the `context.tensors` borrow ends before we touch
            // `context.handles`.
            let output = context
                .tensors
                .get(&output_id)
                .map(|global| (global.id, global.shape.clone(), global.dtype));
            if let Some((fusion_id, shape, dtype)) = output {
                let concrete_id = client.create_empty_handle();
                tensors.push((output_id, concrete_id));
                let handle = RouterTensor::new(concrete_id, shape, dtype, client.clone());
                context.handles.register_handle(fusion_id, handle);
            }
        }

        // Dense shape-dim table indexed by relative dim id (dim ids are dense `0..N`).
        let mut shapes = vec![0usize; context.shapes_relative2global.len()];
        for (relative, concrete) in context.shapes_relative2global.iter() {
            if *relative < shapes.len() {
                shapes[*relative] = *concrete;
            }
        }

        // Concrete scalar values, indexed by their placeholder id.
        let mut scalars = vec![ScalarIr::UInt(0); context.scalars.len()];
        for (scalar_id, value) in context.scalars.iter() {
            let idx = scalar_id.value as usize;
            if idx < scalars.len() {
                scalars[idx] = *value;
            }
        }

        // Concrete slice ranges, indexed by their placeholder id (carried in a relative range's
        // `start`). Cheap to clone — a few `Slice`s per slice op — and, like scalars, they can
        // change between invocations of the same cached graph, so they travel every time.
        let ranges = context.ranges.clone();

        let bindings = GraphBindings {
            tensors,
            shapes,
            scalars,
            ranges,
        };
        match self.graph_id {
            // Already registered: replay by id, sending only the bindings.
            Some(id) => client.execute_graph(id, bindings),
            // First invocation: register the relative graph and execute it in a single round-trip.
            None => {
                let id = next_graph_id();
                self.graph_id = Some(id);
                client.register_and_execute_graph(id, self.graph.clone(), bindings);
            }
        };
    }

    fn to_state(&self) -> RouterGraphExecutionState {
        RouterGraphExecutionState {
            graph: self.graph.clone(),
        }
    }

    fn from_state(device: &R::Device, state: RouterGraphExecutionState) -> Self {
        Self::new(state.graph, device.clone())
    }
}
