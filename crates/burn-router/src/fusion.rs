//! Generic client-side fusion for any [router backend](crate::BackendRouter).
//!
//! Wrapping a router backend as [`Fusion`](burn_fusion::Fusion) turns recurring groups of tensor
//! operations into reusable, client-cached "optimizations". A single greedy [`RouterFuser`]
//! accumulates every operation and is drained only at a sync point, so one optimization covers each
//! connected block between syncs. On execution the group is registered once on the backend (via the
//! [`RouterClient`]) and thereafter invoked by id with only the changing bindings — for the remote
//! backend this means a recurring computation (e.g. a model block) crosses the network once instead
//! of every step.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use burn_backend::DType;
use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_fusion::stream::{Context, OrderedExecution};
use burn_fusion::{
    FuserProperties, FuserStatus, FusionBackend, FusionRuntime, NumOperations, OperationFuser,
    Optimization,
};
use burn_ir::{
    BackendIr, OperationIr, OptimizationBindings, OptimizationId, ScalarIr, TensorHandle, TensorId,
    TensorIr, TensorStatus,
};
use serde::{Deserialize, Serialize};

use crate::{BackendRouter, RouterChannel, RouterClient, RouterTensor, get_client};

static OPT_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_optimization_id() -> OptimizationId {
    OptimizationId(OPT_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

// The router backend already implements `Backend`; these two impls add the `BackendIr` +
// `FusionBackend` glue so it can be wrapped as `Fusion<BackendRouter<R>>`. All four tensor
// primitives are `RouterTensor`, so the handle conversions are the identity (mirroring the
// `impl BackendIr for Fusion<B>` in burn-fusion).
impl<R: RouterChannel> BackendIr for BackendRouter<R> {
    type Handle = RouterTensor<R::Client>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        handle.handle
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        handle.handle
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        handle.handle
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        handle.handle
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor
    }
}

impl<R: RouterChannel> FusionBackend for BackendRouter<R> {
    type FusionRuntime = RouterFusionRuntime<R>;
    type FullPrecisionBackend = Self;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        Self::float_cast(tensor, dtype.into())
    }
}

/// The [fusion runtime](FusionRuntime) for a [router backend](BackendRouter).
///
/// Its [handle](FusionRuntime::FusionHandle) is a [`RouterTensor`] — a lightweight reference to a
/// backend-resident tensor plus the client to reach it — and its optimization is a backend-cached
/// op-graph (see [`RouterOptimization`]).
pub struct RouterFusionRuntime<R: RouterChannel> {
    _p: PhantomData<R>,
}

impl<R: RouterChannel> core::fmt::Debug for RouterFusionRuntime<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RouterFusionRuntime")
    }
}

impl<R: RouterChannel> FusionRuntime for RouterFusionRuntime<R> {
    type OptimizationState = RouterOptimizationState;
    type Optimization = RouterOptimization<R>;
    type FusionHandle = RouterTensor<R::Client>;
    type FusionDevice = R::Device;

    fn fusers(device: R::Device) -> Vec<Box<dyn OperationFuser<Self::Optimization>>> {
        vec![Box::new(RouterFuser::<R>::new(device))]
    }
}

/// A greedy operation fuser that records every operation and never closes itself.
///
/// Because [`status`](OperationFuser::status) always reports [`FuserStatus::Open`], the fusion
/// engine keeps deferring in lazy mode and only drains the queue at a sync point (a read, `sync`,
/// or `flush`). At that point [`finish`](OperationFuser::finish) yields a single
/// [`RouterOptimization`] covering the whole accumulated (connected) block.
pub struct RouterFuser<R: RouterChannel> {
    device: R::Device,
    ops: Vec<OperationIr>,
}

impl<R: RouterChannel> RouterFuser<R> {
    fn new(device: R::Device) -> Self {
        Self {
            device,
            ops: Vec::new(),
        }
    }
}

impl<R: RouterChannel> Clone for RouterFuser<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            ops: self.ops.clone(),
        }
    }
}

impl<R: RouterChannel> OperationFuser<RouterOptimization<R>> for RouterFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        self.ops.push(operation.clone());
    }

    fn finish(&mut self) -> RouterOptimization<R> {
        RouterOptimization::new(self.ops.clone(), self.device.clone())
    }

    fn reset(&mut self) {
        self.ops.clear();
    }

    fn status(&self) -> FuserStatus {
        FuserStatus::Open
    }

    fn properties(&self) -> FuserProperties {
        FuserProperties {
            score: self.ops.len() as u64,
            ready: !self.ops.is_empty(),
        }
    }

    fn len(&self) -> usize {
        self.ops.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<RouterOptimization<R>>> {
        Box::new(self.clone())
    }
}

/// A reusable group of operations, registered once on the backend and thereafter invoked by id.
///
/// The recorded [`graph`](Self::graph) is in *relative* form (positional tensor ids, relative
/// shape dims, scalar placeholders), which is invariant across invocations — that is what lets the
/// backend cache it and the client reuse it. On each [`execute`](Optimization::execute) only the
/// concrete bindings are computed from the [`Context`] and sent; the (large) graph itself travels
/// only on the first invocation.
pub struct RouterOptimization<R: RouterChannel> {
    graph: Vec<OperationIr>,
    device: R::Device,
    /// Backend-side id, assigned and registered on the first execution and reused afterwards.
    server_id: Option<OptimizationId>,
    _p: PhantomData<R>,
}

/// Serializable state for a [`RouterOptimization`].
///
/// The backend id is intentionally not serialized — it is per-connection state, so a deserialized
/// optimization re-registers itself on first use.
#[derive(Serialize, Deserialize)]
pub struct RouterOptimizationState {
    graph: Vec<OperationIr>,
}

impl<R: RouterChannel> RouterOptimization<R> {
    fn new(graph: Vec<OperationIr>, device: R::Device) -> Self {
        Self {
            graph,
            device,
            server_id: None,
            _p: PhantomData,
        }
    }
}

impl<R: RouterChannel> core::fmt::Debug for RouterOptimization<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RouterOptimization")
            .field("len", &self.graph.len())
            .finish()
    }
}

impl<R: RouterChannel> NumOperations for RouterOptimization<R> {
    fn len(&self) -> usize {
        self.graph.len()
    }

    fn name(&self) -> &'static str {
        "RouterOptimization"
    }
}

impl<R: RouterChannel> Optimization<RouterFusionRuntime<R>> for RouterOptimization<R> {
    fn execute(
        &mut self,
        context: &mut Context<RouterTensor<R::Client>>,
        _execution: &OrderedExecution<RouterFusionRuntime<R>>,
    ) {
        // Classify the relative graph's tensors. A tensor is "produced" if it is an output of some
        // op; it is consumed/freed within the block if it appears anywhere with `ReadWrite` status
        // or has an explicit `Drop`. A produced tensor that is neither is a surviving output we
        // must register a handle for. This mirrors the engine's own `drain_queue` freeing logic, so
        // the client-side handle state stays consistent with the unfused path.
        let mut produced: HashSet<TensorId> = HashSet::new();
        let mut read_write: HashSet<TensorId> = HashSet::new();
        let mut dropped: HashSet<TensorId> = HashSet::new();
        for op in &self.graph {
            if let OperationIr::Drop(tensor) = op {
                dropped.insert(tensor.id);
            }
            for tensor in op.outputs() {
                produced.insert(tensor.id);
            }
            for tensor in op.nodes() {
                if tensor.status == TensorStatus::ReadWrite {
                    read_write.insert(tensor.id);
                }
            }
        }

        let client = get_client::<R>(&self.device);

        // Build the relative -> concrete (backend-id + global-shape) bindings for every tensor the
        // graph references, and collect the surviving outputs.
        let mut tensors: Vec<(TensorId, TensorIr)> = Vec::with_capacity(context.tensors.len());
        // (fusion-global id used as the handle-container key, concrete tensor).
        let mut outputs: Vec<(TensorId, TensorIr)> = Vec::new();
        for (relative_id, global) in context.tensors.iter() {
            let concrete_id = match context.handles.get_handle_ref(&global.id) {
                // Already on the backend (an input to this graph): reuse its concrete id.
                Some(handle) => handle.id(),
                // Produced by this graph: allocate a fresh id the backend registers the result under.
                None => client.create_empty_handle(),
            };
            let concrete = TensorIr {
                id: concrete_id,
                shape: global.shape.clone(),
                status: global.status,
                dtype: global.dtype,
            };
            tensors.push((*relative_id, concrete.clone()));

            let survives = produced.contains(relative_id)
                && !read_write.contains(relative_id)
                && !dropped.contains(relative_id);
            if survives {
                outputs.push((global.id, concrete));
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

        // Register lightweight handles for surviving outputs so later ops / reads resolve them.
        for (fusion_id, concrete) in outputs {
            let handle =
                RouterTensor::new(concrete.id, concrete.shape, concrete.dtype, client.clone());
            context.handles.register_handle(fusion_id, handle);
        }

        // Register the relative graph once (first invocation only), then invoke it by id.
        let bindings = OptimizationBindings { tensors, scalars };
        let id = match self.server_id {
            Some(id) => id,
            None => {
                let id = next_optimization_id();
                self.server_id = Some(id);
                client.register_optimization(id, self.graph.clone());
                id
            }
        };
        client.execute_optimization(id, bindings);
    }

    fn to_state(&self) -> RouterOptimizationState {
        RouterOptimizationState {
            graph: self.graph.clone(),
        }
    }

    fn from_state(device: &R::Device, state: RouterOptimizationState) -> Self {
        Self::new(state.graph, device.clone())
    }
}
