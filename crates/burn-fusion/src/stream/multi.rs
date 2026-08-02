use super::{
    StreamId,
    execution::{ExecutionMode, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{FusionRuntime, UnfusedOp, search::BlockOptimization};
use burn_ir::{HandleContainer, OperationIr, TensorId};
use hashbrown::{HashMap, HashSet};

/// Keep track of multiple concurrent lazy streams of operations.
///
/// # Why this exists
///
/// Each `Stream` holds a lazy queue of [`OperationIr`]s whose inputs are assumed
/// to live on that stream. That makes single-stream execution simple — every
/// `TensorId` in a queue is resolvable from the same handle map and the same
/// pending op chain. But a [`FusionTensor`](crate::FusionTensor) is `Send + Clone`,
/// so user code can move or clone a tensor from one thread (= one [`StreamId`]) to
/// another. The receiving thread will then submit ops whose inputs reference a
/// tensor whose home is a *different* stream's queue. This struct is what makes
/// that case behave correctly without giving up the stream-locality invariant.
///
/// # Strategy: shared views
///
/// We never let a foreign-stream tensor id appear in another stream's queue.
/// Instead, when [`FusionTensor::clone`](crate::FusionTensor::clone) or
/// [`FusionTensor::into_ir`](crate::FusionTensor::into_ir) detects that
/// `self.stream != StreamId::current()`, it allocates a fresh id (`dst`) and calls
/// `tag_shared_view` with `(src_stream, src, dst)`. That call does two
/// things, in order:
///
/// 1. **Materialise `src`.** The id `src` might be the output of an op still
///    pending on `src_stream`. We need its backing handle to actually exist before
///    we can alias it. If [`HandleContainer::get_handle_ref`] returns `None` —
///    meaning no op has produced a handle for `src` yet — we drain `src_stream`
///    synchronously, forcing every pending op to run (and thus the handle to be
///    registered). We also record `src` in `shared_sources` so that any
///    *next* share of the same `src` can skip the drain: once registered, a
///    handle stays put (see invariants below).
///
/// 2. **Alias the handle under `dst`.** [`HandleContainer::register_handle`] is
///    called with `dst` and a `clone()` of `src`'s backend handle. Cubecl handles
///    are `Arc`-style reference counters over a backing buffer, so `clone()` is
///    cheap and the buffer survives until the last alias drops. After this call,
///    `handles[src]` and `handles[dst]` are two distinct map entries that both
///    point at the same allocation.
///
/// `shared_view` then returns a new `FusionTensor` carrying `(id = dst, stream =
/// current)`. Every subsequent op on that tensor enqueues on `current` like any
/// other local tensor — the rest of the fusion engine sees no special case.
///
/// # Freeing
///
/// Each `FusionTensor::drop` enqueues an `OperationIr::Drop(ir)` on **its own**
/// `stream` field (the home stream of that particular alias), not the calling
/// thread's stream. So:
///
/// - The original tensor's drop targets `src_stream` and removes `handles[src]`.
/// - The alias tensor's drop targets the stream that minted it and removes
///   `handles[dst]`.
///
/// Each removal decrements the backend handle's `Arc` refcount; the underlying
/// buffer is freed only after the last side drops. No cross-stream coordination
/// is needed.
///
/// # Bounding `shared_sources`
///
/// Naively the set would grow forever, since `tag_shared_view` only ever inserts.
/// Cleanup happens in `register`: as soon as we see an
/// `OperationIr::Drop(ir)` come through, we remove `ir.id` from
/// `shared_sources` immediately — without waiting for the queued `Drop`
/// to actually execute. This is safe because a `Drop` op is registered only
/// after the last live `FusionTensor` with that id has been dropped, so no
/// future `tag_shared_view` can possibly receive that id as a `src`. Removing
/// the entry therefore cannot trigger a redundant drain on any subsequent call.
///
/// # The SSA-like invariant
///
/// Skipping the drain on subsequent shares of the same `src` relies on a
/// property of the fusion IR: **every op output uses a fresh `TensorId`
/// allocated by [`crate::Client::create_empty_handle`], never the id of an
/// input.** Once `handles[src]` is set, no later op overwrites it; the data
/// behind `src` is effectively immutable from the IR's point of view.
///
/// The cubecl-fusion engine *does* sometimes reuse the backing buffer of an
/// input for an output (in-place fusion), but that path is gated by
/// `handle.can_mut()`, which returns false the moment another reference exists.
/// Calling `handle.clone()` in step 2 above is precisely that extra reference,
/// so aliased sources are never eligible for in-place reuse — the engine
/// allocates a fresh output buffer instead.
///
/// # The chained-share fast path
///
/// When a share is itself re-shared (owner → peer → grandpeer), the second
/// `tag_shared_view` call has `src = peer's id`. That id was set up directly by
/// the previous call (via `register_handle`), not by an op enqueued on the peer
/// stream, so `handles.get_handle_ref(&src)` is already `Some` the moment we
/// look. The drain check therefore short-circuits — even though `peer's id` was
/// never added to `shared_sources` (only sources that *required* a
/// drain are tracked there), the handle-existence test alone is sufficient.
pub struct MultiStream<R: FusionRuntime> {
    /// Tensor ids that have been the source of a cross-stream share *and*
    /// required a drain when first shared. Used by `tag_shared_view` to
    /// skip the drain on subsequent shares of the same source. Bounded by
    /// pruning in `register` when a `Drop` op for the id is enqueued —
    /// see the struct-level docs for the full strategy.
    shared_sources: HashSet<TensorId>,
    streams: HashMap<StreamId, Stream<R>>,
    optimizations: ExecutionPlanStore<R::Optimization>,
    device: R::FusionDevice,
    #[cfg(feature = "memory-checks")]
    memory_checks: super::memory_checks::MemoryChecks,
}

impl<R: FusionRuntime> MultiStream<R> {
    pub(crate) fn new(device: R::FusionDevice) -> Self {
        Self {
            shared_sources: HashSet::new(),
            streams: HashMap::new(),
            optimizations: ExecutionPlanStore::new(),
            device,
            #[cfg(feature = "memory-checks")]
            memory_checks: super::memory_checks::MemoryChecks::default(),
        }
    }

    /// Register a new tensor operation on the given `stream`.
    pub(crate) fn register(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        // Bound `shared_sources` (see struct-level docs). When the last `FusionTensor`
        // for an id is dropped, a `Drop` op is registered here. At that point no live
        // `FusionTensor` holds this id, so no future `tag_shared_view` can use it as
        // a source — it is safe to drop the entry immediately, without waiting for
        // the queued `Drop` op to actually execute.
        if let OperationIr::Drop(ir) = &repr {
            self.shared_sources.remove(&ir.id);
        }

        self.enqueue_operation(stream, repr, operation, handles);

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(stream, handles.handle_ids().copied());
    }

    /// Set up a cross-stream alias `dst` for the foreign tensor `src` that lives on
    /// `src_stream`. Called when [`FusionTensor::clone`](crate::FusionTensor::clone)
    /// or [`FusionTensor::into_ir`](crate::FusionTensor::into_ir) detects that the
    /// tensor's home stream is not the current stream.
    ///
    /// See the [`MultiStream`] struct-level docs for the full strategy. In short:
    ///
    /// - If `src`'s handle isn't materialised yet, drain `src_stream` so the
    ///   producing op runs and registers it. Skip the drain on subsequent shares
    ///   of the same source by remembering it in `shared_sources` *or* by
    ///   observing that the handle is already in the container (which covers the
    ///   chained-share case where `src` is itself a previously-aliased view).
    /// - Then alias the backing handle under `dst`. `register_handle` clones the
    ///   cubecl handle (`Arc`-style), so both ids share refcount on the buffer
    ///   until each side's own `Drop` op runs.
    pub fn tag_shared_view(
        &mut self,
        src_stream: StreamId,
        src: TensorId,
        dst: TensorId,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        // Drain only when neither short-circuit applies: `shared_sources` records ids
        // we already drained for, and a `Some` handle means `src` is materialised
        // (e.g., it was itself set up by an earlier `tag_shared_view` call). We
        // record `src` only when we actually drain — the handle-existence path is
        // naturally idempotent on later calls.
        if !self.shared_sources.contains(&src) && handles.get_handle_ref(&src).is_none() {
            self.shared_sources.insert(src);
            self.drain(handles, src_stream);
        }

        if let Some(handle) = handles.get_handle_ref(&src) {
            // Not a bare `clone()`: remote backends need a fresh server-side handle over the same
            // buffer so consuming one alias doesn't free it for the other stream. Local backends'
            // `alias_handle` default *is* `clone()`. See `FusionRuntime::alias_handle`.
            let alias = R::alias_handle(handle);
            handles.register_handle(dst, alias);
        }
    }

    /// Enqueue an operation on the queue for `stream` and run the lazy processor.
    fn enqueue_operation(
        &mut self,
        stream: StreamId,
        repr: OperationIr,
        operation: UnfusedOp<R>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let s = self
            .streams
            .entry(stream)
            .or_insert_with(|| Stream::new(self.device.clone()));
        s.queue.add(repr, operation);

        let len_before = s.queue.global.len();
        s.processor.process(
            Segment::new(&mut s.queue, handles, stream),
            &mut self.optimizations,
            ExecutionMode::Lazy,
        );
        let len_after = s.queue.global.len();
        s.cursor += (len_before - len_after) as u64;
    }

    /// Mark a tensor as read.
    #[allow(unused_variables)]
    pub fn mark_read(
        &mut self,
        id: StreamId,
        ir: &burn_ir::TensorIr,
        handles: &HandleContainer<R::FusionHandle>,
    ) {
        if !matches!(ir.status, burn_ir::TensorStatus::ReadWrite) {
            return;
        };

        let stream = match self.streams.get_mut(&id) {
            Some(val) => val,
            None => return,
        };

        stream.queue.variables.remove(&ir.id);

        if stream.queue.variables.is_empty() {
            self.streams.remove(&id);
        }

        #[cfg(feature = "memory-checks")]
        self.memory_checks.check(&self.streams, handles);
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(id, handles.handle_ids().copied());
    }

    /// Drain a stream.
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        id.executes(|| {
            if let Some(stream) = self.streams.get_mut(&id) {
                let num_executed = stream.queue.global.len();
                stream.processor.process(
                    Segment::new(&mut stream.queue, handles, id),
                    &mut self.optimizations,
                    ExecutionMode::Sync,
                );
                stream.cursor += num_executed as u64;
                // A drain is a boundary even when the queue was already empty.
                stream.queue.flush_deferred(handles);
            }
        });
        #[cfg(feature = "test-util")]
        crate::inspect::emit_handle_snapshot(id, handles.handle_ids().copied());
    }

    /// How a cross-thread read of `ir` on `id`'s stream must be served — see
    /// [`ReadPlan`].
    pub(crate) fn read_plan(
        &self,
        id: StreamId,
        ir: &burn_ir::TensorIr,
        handles: &HandleContainer<R::FusionHandle>,
    ) -> ReadPlan {
        if handles.get_handle_ref(&ir.id).is_none() {
            // Only the queue can order the read after the pending producer.
            return ReadPlan::Drain;
        }
        if !matches!(ir.status, burn_ir::TensorStatus::ReadWrite) {
            return ReadPlan::Direct;
        }
        // Defer the last-use free only while pending ops still reference the
        // tensor — they also guarantee a boundary to release it.
        match self.streams.get(&id) {
            Some(stream) if stream.queue.references_tensor(ir.id) => ReadPlan::DeferFree,
            _ => ReadPlan::Direct,
        }
    }

    /// Queue `ir`'s last-use free to run at `id`'s next execution boundary.
    pub(crate) fn defer_free(&mut self, id: StreamId, ir: burn_ir::TensorIr) {
        if let Some(stream) = self.streams.get_mut(&id) {
            stream.queue.deferred_frees.push(ir);
        }
    }

    /// A cross-thread `Drop` of a materialized tensor: free it without
    /// touching the queue — immediately, or at the next execution boundary
    /// while pending ops still reference it. Returns `false` when the handle
    /// does not exist yet; the caller must fall back to the queue.
    pub(crate) fn foreign_drop(
        &mut self,
        id: StreamId,
        ir: burn_ir::TensorIr,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> bool {
        if handles.get_handle_ref(&ir.id).is_none() {
            return false;
        }
        // Mirrors `register`'s bookkeeping for `Drop` ops.
        self.shared_sources.remove(&ir.id);
        match self.streams.get_mut(&id) {
            Some(stream) if stream.queue.references_tensor(ir.id) => {
                stream.queue.deferred_frees.push(ir);
            }
            Some(stream) => {
                stream.queue.variables.remove(&ir.id);
                handles.free(&ir);
                if stream.queue.variables.is_empty() {
                    self.streams.remove(&id);
                }
            }
            None => handles.free(&ir),
        }
        true
    }
}

/// How a cross-thread read of a tensor on another stream must be served.
///
/// A cross-thread event lands at an arbitrary point between the home
/// thread's registrations; draining there would compile different fused
/// blocks run to run for the same op sequence (seen as autotune-key churn).
/// A drain is forced only when the tensor's handle does not exist yet.
pub(crate) enum ReadPlan {
    /// A pending operation still has to produce the tensor: drain first.
    Drain,
    /// Read (and, for a last use, free) directly — nothing pending is
    /// involved.
    Direct,
    /// Read through a `ReadOnly` view; the last-use free runs at the next
    /// execution boundary, guaranteed to come since a pending op still
    /// references the tensor.
    DeferFree,
}

pub(crate) struct Stream<R: FusionRuntime> {
    pub(crate) queue: OperationQueue<R>,
    processor: Processor<R::Optimization>,
    pub(crate) cursor: u64,
}

#[derive(new)]
struct Segment<'a, R: FusionRuntime> {
    queue: &'a mut OperationQueue<R>,
    handles: &'a mut HandleContainer<R::FusionHandle>,
    id: StreamId,
}

impl<R: FusionRuntime> StreamSegment<R::Optimization> for Segment<'_, R> {
    fn operations(&self) -> &[OperationIr] {
        &self.queue.relative
    }

    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<R::Optimization>) {
        self.queue.execute(id, self.handles, store, self.id)
    }

    fn execute_unfused(&mut self, optimization: BlockOptimization<R::Optimization>) {
        self.queue
            .execute_unfused(optimization, self.handles, self.id)
    }
}

impl<R: FusionRuntime> Stream<R> {
    fn new(device: R::FusionDevice) -> Self {
        Self {
            processor: Processor::new(R::fusers(device)),
            queue: OperationQueue::new(),
            cursor: 0,
        }
    }
}

/// Cross-thread last-use (consuming read or foreign `Drop`) of a materialized
/// tensor. The core property is determinism: the composition must depend on
/// the home thread's op sequence alone, never on when the foreign message
/// lands (the `*_timing_*` tests). The rest pin the free-timing contract:
/// deferred only while pending ops still reference the tensor, released at
/// the next boundary, immediate otherwise.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FuserProperties, FuserStatus, NumOperations, OperationFuser, Optimization, UnfusedOp,
        stream::{Context, Operation, OrderedExecution},
    };
    use burn_backend::{DType, DeviceId, DeviceOps, DeviceSettings, Shape};
    use burn_ir::{FloatOperationIr, TensorIr, TensorStatus, UnaryOpIr};
    use burn_std::{BoolDType, FloatDType, IntDType, device::Device};

    #[derive(Debug)]
    struct TestRuntime;

    #[derive(Clone, Debug, Default, PartialEq)]
    struct TestDevice;

    impl Device for TestDevice {
        fn from_id(_device_id: DeviceId) -> Self {
            Self
        }

        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: 0,
            }
        }
    }

    impl DeviceOps for TestDevice {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::with_dtypes(FloatDType::F32, IntDType::I32, BoolDType::Native)
        }
    }

    #[derive(Clone, Debug)]
    struct TestHandle;

    #[derive(Debug)]
    struct TestOptimization;

    impl NumOperations for TestOptimization {
        fn len(&self) -> usize {
            0
        }

        fn name(&self) -> &'static str {
            "TestOptimization"
        }
    }

    impl Optimization<TestRuntime> for TestOptimization {
        fn execute(
            &mut self,
            _context: &mut Context<TestHandle>,
            _execution: &OrderedExecution<TestRuntime>,
        ) {
        }

        fn to_state(&self) {}

        fn from_state(_device: &TestDevice, _state: ()) -> Self {
            Self
        }
    }

    /// Stays open and never ready, so registered operations accumulate until
    /// an explicit drain.
    #[derive(Clone, Debug, Default)]
    struct NeverReadyFuser {
        count: usize,
    }

    impl OperationFuser<TestOptimization> for NeverReadyFuser {
        fn fuse(&mut self, _operation: &OperationIr) {
            self.count += 1;
        }

        fn finish(&mut self) -> TestOptimization {
            TestOptimization
        }

        fn reset(&mut self) {
            self.count = 0;
        }

        fn status(&self) -> FuserStatus {
            FuserStatus::Open
        }

        fn properties(&self) -> FuserProperties {
            FuserProperties {
                score: 0,
                ready: false,
            }
        }

        fn len(&self) -> usize {
            self.count
        }

        fn clone_dyn(&self) -> Box<dyn OperationFuser<TestOptimization>> {
            Box::new(self.clone())
        }
    }

    impl FusionRuntime for TestRuntime {
        type OptimizationState = ();
        type Optimization = TestOptimization;
        type FusionHandle = TestHandle;
        type FusionDevice = TestDevice;

        fn fusers(_device: TestDevice) -> Vec<Box<dyn OperationFuser<TestOptimization>>> {
            vec![Box::new(NeverReadyFuser::default())]
        }
    }

    /// Registers the output handle of [`exp_op`] when executed.
    #[derive(Debug)]
    struct ProduceOp {
        out: TensorId,
    }

    impl Operation<TestRuntime> for ProduceOp {
        fn execute(&self, handles: &mut HandleContainer<TestHandle>) {
            handles.register_handle(self.out, TestHandle);
        }
    }

    fn tensor_ir(id: TensorId, status: TensorStatus) -> TensorIr {
        TensorIr {
            id,
            shape: Shape::new([32, 32]),
            status,
            dtype: DType::F32,
        }
    }

    fn exp_op(input: TensorId, out: TensorId) -> OperationIr {
        OperationIr::Float(
            DType::F32,
            FloatOperationIr::Exp(UnaryOpIr {
                input: tensor_ir(input, TensorStatus::ReadOnly),
                out: tensor_ir(out, TensorStatus::NotInit),
            }),
        )
    }

    struct TestSetup {
        streams: MultiStream<TestRuntime>,
        handles: HandleContainer<TestHandle>,
        id: StreamId,
    }

    impl TestSetup {
        fn new() -> Self {
            Self {
                streams: MultiStream::new(TestDevice),
                handles: HandleContainer::new(),
                id: StreamId::current(),
            }
        }

        fn register_exp(&mut self, input: TensorId, out: TensorId) {
            self.streams.register(
                self.id,
                exp_op(input, out),
                UnfusedOp::new(ProduceOp { out }, self.id),
                &mut self.handles,
            );
        }

        fn num_pending(&self) -> usize {
            self.streams
                .streams
                .get(&self.id)
                .map(|stream| stream.queue.global.len())
                .unwrap_or(0)
        }

        /// The relative IR sequence that composition and autotune keys
        /// derive from.
        fn composition(&self) -> Vec<OperationIr> {
            self.streams
                .streams
                .get(&self.id)
                .map(|stream| stream.queue.relative.clone())
                .unwrap_or_default()
        }

        /// Mimic the server's `prepare_read` for a consuming read of a
        /// materialized tensor.
        fn consuming_read(&mut self, tensor: TensorId) {
            let ir = tensor_ir(tensor, TensorStatus::ReadWrite);
            match self.streams.read_plan(self.id, &ir, &self.handles) {
                ReadPlan::Drain => panic!("materialized tensor must not force a drain"),
                ReadPlan::Direct => {
                    self.handles.free(&ir);
                    self.streams.mark_read(self.id, &ir, &self.handles);
                }
                ReadPlan::DeferFree => self.streams.defer_free(self.id, ir),
            }
        }
    }

    /// Run the op sequence with a foreign event injected before the
    /// `inject_at`-th registration (after all of them when `inject_at ==
    /// ops.len()`) and return the composition.
    fn compose_with_injection(
        shared: TensorId,
        ops: &[(u64, u64)],
        inject_at: Option<usize>,
        inject: impl Fn(&mut TestSetup),
    ) -> Vec<OperationIr> {
        let mut setup = TestSetup::new();
        setup.handles.register_handle(shared, TestHandle);
        for (i, (input, out)) in ops.iter().enumerate() {
            if inject_at == Some(i) {
                inject(&mut setup);
            }
            setup.register_exp(TensorId::new(*input), TensorId::new(*out));
        }
        if inject_at == Some(ops.len()) {
            inject(&mut setup);
        }
        setup.composition()
    }

    /// The composition must not depend on where a foreign `Drop` lands
    /// between home registrations: cutting the segment or enqueuing the drop
    /// would compile a different sequence per run.
    #[test]
    fn foreign_drop_timing_does_not_change_composition() {
        let shared = TensorId::new(100);
        let drop_shared = |setup: &mut TestSetup| {
            let handled = setup.streams.foreign_drop(
                setup.id,
                tensor_ir(shared, TensorStatus::ReadWrite),
                &mut setup.handles,
            );
            assert!(
                handled,
                "materialized tensor must not fall back to the queue"
            );
        };

        // The pending ops never reference the shared tensor.
        let ops = [(0, 1), (1, 2), (2, 3)];
        let baseline = compose_with_injection(shared, &ops, None, drop_shared);
        assert_eq!(baseline.len(), 3, "nothing may cut the pending segment");
        for at in 0..=ops.len() {
            let composition = compose_with_injection(shared, &ops, Some(at), drop_shared);
            assert_eq!(composition, baseline, "drop injected before op {at}");
        }

        // The first pending op reads the shared tensor; the drop can only
        // arrive after that registration.
        let ops = [(100, 1), (1, 2), (2, 3)];
        let baseline = compose_with_injection(shared, &ops, None, drop_shared);
        assert_eq!(baseline.len(), 3, "nothing may cut the pending segment");
        for at in 1..=ops.len() {
            let composition = compose_with_injection(shared, &ops, Some(at), drop_shared);
            assert_eq!(composition, baseline, "drop injected before op {at}");
        }
    }

    /// Same property for a reader thread resolving a materialized value.
    #[test]
    fn consuming_read_timing_does_not_change_composition() {
        let shared = TensorId::new(100);
        let read_shared = |setup: &mut TestSetup| setup.consuming_read(shared);

        // The pending ops never reference the shared tensor.
        let ops = [(0, 1), (1, 2), (2, 3)];
        let baseline = compose_with_injection(shared, &ops, None, read_shared);
        assert_eq!(baseline.len(), 3, "nothing may cut the pending segment");
        for at in 0..=ops.len() {
            let composition = compose_with_injection(shared, &ops, Some(at), read_shared);
            assert_eq!(composition, baseline, "read injected before op {at}");
        }

        // The first pending op reads the shared tensor.
        let ops = [(100, 1), (1, 2), (2, 3)];
        let baseline = compose_with_injection(shared, &ops, None, read_shared);
        assert_eq!(baseline.len(), 3, "nothing may cut the pending segment");
        for at in 1..=ops.len() {
            let composition = compose_with_injection(shared, &ops, Some(at), read_shared);
            assert_eq!(composition, baseline, "read injected before op {at}");
        }
    }

    #[test]
    fn read_of_unmaterialized_tensor_must_drain() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.register_exp(t0, t1);

        // t1's producer is still pending: only the queue can order the read.
        let plan = setup.streams.read_plan(
            setup.id,
            &tensor_ir(t1, TensorStatus::ReadWrite),
            &setup.handles,
        );
        assert!(matches!(plan, ReadPlan::Drain));
    }

    #[test]
    fn read_of_referenced_tensor_defers_free_and_keeps_composition() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        assert_eq!(setup.num_pending(), 1);

        // Materialized, but a pending op still reads it: the free must wait.
        let ir = tensor_ir(t0, TensorStatus::ReadWrite);
        let plan = setup.streams.read_plan(setup.id, &ir, &setup.handles);
        assert!(matches!(plan, ReadPlan::DeferFree));

        setup.streams.defer_free(setup.id, ir);
        assert!(
            setup.handles.has_handle(&t0),
            "still readable by the kernel"
        );
        assert_eq!(setup.num_pending(), 1, "composition must not be cut");

        setup.streams.drain(&mut setup.handles, setup.id);
        assert!(setup.handles.has_handle(&t1), "producer ran");
        assert!(!setup.handles.has_handle(&t0), "freed at the boundary");
    }

    #[test]
    fn read_of_unreferenced_tensor_frees_directly_even_on_idle_stream() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        setup.streams.drain(&mut setup.handles, setup.id);
        assert_eq!(setup.num_pending(), 0);

        // Queue empty, only a stale `variables` entry remains for t0:
        // deferring could park the free forever, so free directly.
        let plan = setup.streams.read_plan(
            setup.id,
            &tensor_ir(t0, TensorStatus::ReadWrite),
            &setup.handles,
        );
        assert!(matches!(plan, ReadPlan::Direct));
    }

    #[test]
    fn foreign_drop_of_unmaterialized_tensor_falls_back_to_the_queue() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.register_exp(t0, t1);

        // t1's producer is pending: only the queue can order the drop.
        let handled = setup.streams.foreign_drop(
            setup.id,
            tensor_ir(t1, TensorStatus::ReadWrite),
            &mut setup.handles,
        );
        assert!(!handled);
        assert_eq!(setup.num_pending(), 1);
    }

    #[test]
    fn foreign_drop_of_referenced_tensor_is_deferred_to_the_next_boundary() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);

        let handled = setup.streams.foreign_drop(
            setup.id,
            tensor_ir(t0, TensorStatus::ReadWrite),
            &mut setup.handles,
        );
        assert!(handled);
        assert!(
            setup.handles.has_handle(&t0),
            "a pending op still reads t0: the free must wait for the boundary"
        );
        assert_eq!(setup.num_pending(), 1, "composition must not be cut");

        setup.streams.drain(&mut setup.handles, setup.id);
        assert!(!setup.handles.has_handle(&t0), "freed at the boundary");
    }

    #[test]
    fn foreign_drop_of_unreferenced_tensor_frees_immediately() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        setup.streams.drain(&mut setup.handles, setup.id);

        // Nothing pending references t0 anymore: free on the spot.
        let handled = setup.streams.foreign_drop(
            setup.id,
            tensor_ir(t0, TensorStatus::ReadWrite),
            &mut setup.handles,
        );
        assert!(handled);
        assert!(!setup.handles.has_handle(&t0));

        let stale = setup
            .streams
            .streams
            .get(&setup.id)
            .is_some_and(|stream| stream.queue.variables.contains_key(&t0));
        assert!(!stale, "the stale variables entry is cleaned up");
    }
}
