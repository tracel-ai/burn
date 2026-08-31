use super::{
    StreamId,
    execution::{ExecutionMode, Processor, StreamSegment},
    queue::OperationQueue,
    store::{ExecutionPlanId, ExecutionPlanStore},
};
use crate::{FusionRuntime, UnfusedOp, search::BlockOptimization};
use burn_ir::{ExistingHandle, HandleContainer, OperationIr, TensorId};
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
        // A share of an errored tensor carries the error across with it.
        // Without this the alias would be a plain missing handle on the
        // receiving stream, and the thread that reads it would be told the
        // tensor does not exist rather than why it was never written.
        //
        // Checked before the drain for the same reason `read_plan` returns
        // `Direct`: no pending operation is going to produce `src`, so there
        // is nothing for a drain to order the share after.
        if let Some(error) = handles.error(&src).cloned() {
            handles.set_error(dst, error, ExistingHandle::Displace);
            return;
        }

        // Drain only when neither short-circuit applies: `shared_sources` records ids
        // we already drained for, and a `Some` handle means `src` is materialised
        // (e.g., it was itself set up by an earlier `tag_shared_view` call). We
        // record `src` only when we actually drain — the handle-existence path is
        // naturally idempotent on later calls.
        if !self.shared_sources.contains(&src) && handles.get_handle_ref(&src).is_none() {
            self.drain(handles, src_stream);

            // Mark the source only once the drain actually materialised it. A
            // drain that failed to produce it leaves no handle, and marking
            // `src` anyway would make every later share of it skip the drain
            // too and mint a `dst` with no handle behind it.
            if handles.get_handle_ref(&src).is_some() {
                self.shared_sources.insert(src);
            }

            // The drain is what surfaced the failure: `src` had a pending
            // producer, and that producer did not produce it.
            if let Some(error) = handles.error(&src).cloned() {
                handles.set_error(dst, error, ExistingHandle::Displace);
                return;
            }
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

    /// Run `id`'s pending segment to completion.
    ///
    /// Reports nothing. An operation that fails leaves its error on the
    /// tensors it was going to write (see `execution::set_output_errors`), so
    /// it is delivered by the read of one of *those* tensors — the point where a
    /// caller is actually waiting for that data — and not to whoever happened
    /// to drain the stream next. A drain that shares no tensor with the
    /// failure has nothing to report and returns normally.
    pub fn drain(&mut self, handles: &mut HandleContainer<R::FusionHandle>, id: StreamId) {
        id.executes(|| {
            let Some(stream) = self.streams.get_mut(&id) else {
                return;
            };
            let num_executed = stream.queue.global.len();
            stream.processor.process(
                Segment::new(&mut stream.queue, handles, id),
                &mut self.optimizations,
                ExecutionMode::Sync,
            );
            stream.cursor += num_executed as u64;
            // A drain is a boundary even when the queue was already empty.
            stream.queue.flush_deferred(handles);
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
        if handles.error(&ir.id).is_some() {
            // Errored: no pending operation is going to produce this tensor,
            // so there is nothing for a drain to order the read after. The
            // read itself reports the failure.
            return ReadPlan::Direct;
        }
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
        FuserProperties, FuserStatus, NumOperations, OperationFuser, OperationRan, Optimization,
        UnfusedOp,
        stream::{Context, Operation, OrderedExecution},
    };
    use burn_backend::{DType, DeviceId, DeviceOps, DeviceSettings, Shape};
    use burn_ir::{FloatOperationIr, TensorIr, TensorStatus, UnaryOpIr};
    use burn_std::{BoolDType, FloatDType, IntDType, device::Device};

    #[derive(Debug)]
    struct TestRuntime;

    /// Which fuser [`TestRuntime`] hands to the streams on a device, which is
    /// what decides where an operation actually executes.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum Fusing {
        /// Never closes, so operations accumulate until an explicit drain.
        #[default]
        Deferred,
        /// Closes on its first operation, so the processor commits during the
        /// registration that queued it.
        Eager,
        /// Closes and reports ready, so the block compiles to one fused
        /// kernel and runs through `OrderedExecution::execute_optimization`.
        Fused,
    }

    /// Carried on the device rather than in ambient state because
    /// [`FusionRuntime::fusers`] is handed the device, and a choice that
    /// travels with the value cannot leak into the next test.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct TestDevice {
        fusing: Fusing,
    }

    impl Device for TestDevice {
        fn from_id(device_id: DeviceId) -> Self {
            let fusing = match device_id.index_id {
                0 => Fusing::Deferred,
                1 => Fusing::Eager,
                _ => Fusing::Fused,
            };
            Self { fusing }
        }

        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: match self.fusing {
                    Fusing::Deferred => 0,
                    Fusing::Eager => 1,
                    Fusing::Fused => 2,
                },
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

    /// One fused kernel over the operations a [`FusingFuser`] collected: it
    /// writes the whole block's outputs, or it cannot serve its problem and
    /// writes none of them.
    #[derive(Debug, Default)]
    struct TestOptimization {
        /// How many operations the kernel replaced. What the queue consumes.
        len: usize,
        /// Relative ids of every tensor the block writes, resolved through
        /// `context.tensors` at execution the way a real optimization does.
        outputs: Vec<TensorId>,
        /// Whether the kernel can serve the problem it was compiled for.
        panics: bool,
    }

    impl NumOperations for TestOptimization {
        fn len(&self) -> usize {
            self.len
        }

        fn name(&self) -> &'static str {
            "TestOptimization"
        }
    }

    impl Optimization<TestRuntime> for TestOptimization {
        fn execute(
            &mut self,
            context: &mut Context<TestHandle>,
            _execution: &OrderedExecution<TestRuntime>,
        ) {
            if self.panics {
                panic!("this fused kernel cannot serve its problem");
            }

            for relative in &self.outputs {
                let global = context
                    .tensors
                    .get(relative)
                    .expect("every fused output is in the context")
                    .id;
                context.handles.register_handle(global, TestHandle);
            }
        }

        fn to_state(&self) {}

        fn from_state(_device: &TestDevice, _state: ()) -> Self {
            Self::default()
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
            TestOptimization::default()
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

    /// Closes on its first operation, so the processor commits the segment on
    /// the very registration that queued it. The counterpart of
    /// [`NeverReadyFuser`], covering the path where an operation executes
    /// during a fire-and-forget registration instead of at a drain.
    #[derive(Clone, Debug, Default)]
    struct EagerFuser {
        count: usize,
    }

    impl OperationFuser<TestOptimization> for EagerFuser {
        fn fuse(&mut self, _operation: &OperationIr) {
            self.count += 1;
        }

        fn finish(&mut self) -> TestOptimization {
            TestOptimization::default()
        }

        fn reset(&mut self) {
            self.count = 0;
        }

        fn status(&self) -> FuserStatus {
            FuserStatus::Closed
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

    /// Closes and reports ready, so its block compiles to one fused kernel.
    /// The counterpart of [`NeverReadyFuser`], covering the path where a
    /// segment runs through `OrderedExecution::execute_optimization`.
    ///
    /// The kernel it emits panics when the block contains a [`failing_op`],
    /// the way a real fuser reads the IR to decide what it can emit.
    #[derive(Clone, Debug, Default)]
    struct FusingFuser {
        outputs: Vec<TensorId>,
        len: usize,
        panics: bool,
    }

    impl OperationFuser<TestOptimization> for FusingFuser {
        fn fuse(&mut self, operation: &OperationIr) {
            self.len += 1;
            self.outputs.extend(operation.outputs().map(|node| node.id));
            self.panics |= matches!(operation, OperationIr::Float(_, FloatOperationIr::Log(_)));
        }

        fn finish(&mut self) -> TestOptimization {
            TestOptimization {
                len: self.len,
                outputs: core::mem::take(&mut self.outputs),
                panics: self.panics,
            }
        }

        fn reset(&mut self) {
            *self = Self::default();
        }

        /// Two operations to a block, so a test can tell "the whole write
        /// set" from "the output of the one that failed".
        fn status(&self) -> FuserStatus {
            match self.len >= 2 {
                true => FuserStatus::Closed,
                false => FuserStatus::Open,
            }
        }

        fn properties(&self) -> FuserProperties {
            FuserProperties {
                score: 1,
                ready: self.len > 0,
            }
        }

        fn len(&self) -> usize {
            self.len
        }

        fn clone_dyn(&self) -> Box<dyn OperationFuser<TestOptimization>> {
            Box::new(self.clone())
        }
    }

    thread_local! {
        /// Tensors reclaimed with [`OperationRan::No`], recorded by
        /// [`TestRuntime::free_handle`].
        ///
        /// Ambient state, unlike [`TestDevice`]'s fuser choice, because
        /// `free_handle` is a static method taking neither a device nor a
        /// runtime value — a test has no other channel into it.
        static UNRUN_FREES: std::cell::RefCell<Vec<TensorId>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    impl FusionRuntime for TestRuntime {
        type OptimizationState = ();
        type Optimization = TestOptimization;
        type FusionHandle = TestHandle;
        type FusionDevice = TestDevice;

        fn free_handle(
            handles: &mut HandleContainer<TestHandle>,
            tensor: &TensorIr,
            ran: OperationRan,
        ) {
            if tensor.status == TensorStatus::ReadWrite && ran == OperationRan::No {
                UNRUN_FREES.with(|freed| freed.borrow_mut().push(tensor.id));
            }
            handles.free(tensor);
        }

        fn fusers(device: TestDevice) -> Vec<Box<dyn OperationFuser<TestOptimization>>> {
            match device.fusing {
                Fusing::Deferred => vec![Box::new(NeverReadyFuser::default())],
                Fusing::Eager => vec![Box::new(EagerFuser::default())],
                Fusing::Fused => vec![Box::new(FusingFuser::default())],
            }
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

    /// Registers its output handle and *then* panics, the way in-place
    /// fusion does: the output is aliased to its input while the launch is
    /// planned, so the handle is there before the kernel that fills it runs.
    #[derive(Debug)]
    struct AliasThenPanicOp {
        out: TensorId,
    }

    impl Operation<TestRuntime> for AliasThenPanicOp {
        fn execute(&self, handles: &mut HandleContainer<TestHandle>) {
            handles.register_handle(self.out, TestHandle);
            panic!("this operation cannot serve its problem");
        }
    }

    /// Releases a tensor, the way a `FusionTensor`'s drop does, and records
    /// that it ran.
    ///
    /// Whether it ran is the thing to assert on, not whether the entry is
    /// gone: `drain_queue` frees a `ReadWrite` node itself, so the container
    /// ends up in the same state either way. Backends whose drop does more
    /// than clear that entry — the remote one frees the tensor server-side —
    /// need the operation itself to execute.
    #[derive(Debug)]
    struct DropOp {
        id: TensorId,
        ran: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl Operation<TestRuntime> for DropOp {
        fn execute(&self, handles: &mut HandleContainer<TestHandle>) {
            self.ran.store(true, std::sync::atomic::Ordering::Relaxed);
            handles.remove_handle(self.id);
        }
    }

    /// Panics when executed, the way a pinned kernel that cannot serve its
    /// problem does in the benchmark sweeps downstream.
    #[derive(Debug)]
    struct PanicOp;

    impl Operation<TestRuntime> for PanicOp {
        fn execute(&self, _handles: &mut HandleContainer<TestHandle>) {
            panic!("this operation cannot serve its problem");
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

    /// An operation a [`FusingFuser`] compiles into a kernel that cannot
    /// serve its problem. Distinguished by its IR, the way a real fuser reads
    /// the operations it is given rather than being told out of band.
    fn failing_op(input: TensorId, out: TensorId) -> OperationIr {
        OperationIr::Float(
            DType::F32,
            FloatOperationIr::Log(UnaryOpIr {
                input: tensor_ir(input, TensorStatus::ReadOnly),
                out: tensor_ir(out, TensorStatus::NotInit),
            }),
        )
    }

    /// An operation whose input is its last use, so the drained block
    /// reclaims it through [`FusionRuntime::free_handle`].
    fn consume_op(input: TensorId, out: TensorId) -> OperationIr {
        OperationIr::Float(
            DType::F32,
            FloatOperationIr::Exp(UnaryOpIr {
                input: tensor_ir(input, TensorStatus::ReadWrite),
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
            Self::on(Fusing::Deferred)
        }

        fn on(fusing: Fusing) -> Self {
            Self {
                streams: MultiStream::new(TestDevice { fusing }),
                handles: HandleContainer::new(),
                id: StreamId::current(),
            }
        }

        /// A setup whose streams execute on registration rather than
        /// deferring — the lazy path a fire-and-forget caller takes.
        fn eager() -> Self {
            Self::on(Fusing::Eager)
        }

        /// A setup whose streams compile each segment into one fused kernel.
        fn fused() -> Self {
            Self::on(Fusing::Fused)
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

    /// The error a failure leaves is what a read of the tensor reports. The
    /// stream itself is not poisoned: the queue is intact, the drain returns
    /// normally, and only the data the failure was going to write is gone.
    #[test]
    fn a_failing_operation_errors_the_tensor_it_was_going_to_write() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );

        // A drain reports nothing: the failure lives on the tensor, and the
        // read of that tensor is what surfaces it.
        setup.streams.drain(&mut setup.handles, setup.id);

        let error = setup
            .handles
            .error(&t1)
            .expect("the output holds the error");
        assert!(
            error
                .root()
                .contains("this operation cannot serve its problem"),
            "the error names the panic that caused it: {error}"
        );
        assert!(!setup.handles.has_handle(&t1), "there is no data behind it");

        // Reading it is where a caller finally hears about it.
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            setup
                .handles
                .get_handle(&t1, &burn_ir::TensorStatus::ReadWrite)
        }));
        let payload = read.expect_err("the read must not hand back untouched memory");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(
            message.contains("never written")
                && message.contains("this operation cannot serve its problem"),
            "the read names the root cause: {message}"
        );
    }

    /// The property the whole design is for: a failure is a fact about the
    /// tensors it was going to write, so work that shares none of them is
    /// untouched — even when it was queued on the same stream, behind the
    /// operation that failed.
    #[test]
    fn work_sharing_no_tensor_with_a_failure_still_runs() {
        let mut setup = TestSetup::new();
        let (a0, a1) = (TensorId::new(0), TensorId::new(1));
        let (b0, b1) = (TensorId::new(10), TensorId::new(11));

        setup.handles.register_handle(a0, TestHandle);
        setup.handles.register_handle(b0, TestHandle);

        // Two independent chains queued on one stream; the first one fails.
        setup.streams.register(
            setup.id,
            exp_op(a0, a1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.register_exp(b0, b1);

        setup.streams.drain(&mut setup.handles, setup.id);

        assert!(
            setup.handles.error(&a1).is_some(),
            "the failing chain holds the error"
        );
        assert!(
            setup.handles.has_handle(&b1),
            "the independent chain behind it still ran"
        );
        assert!(setup.handles.error(&b1).is_none(), "and holds no error");
    }

    /// Work downstream of an errored tensor cannot run either — its input was
    /// never written — but it must report the failure that started it, not a
    /// fresh one of its own. However long the chain, the root is the same.
    #[test]
    fn an_error_propagates_downstream_carrying_its_root() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);
        let t2 = TensorId::new(2);
        let t3 = TensorId::new(3);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        // Two operations reading, in turn, what the failure never wrote.
        setup.register_exp(t1, t2);
        setup.register_exp(t2, t3);

        setup.streams.drain(&mut setup.handles, setup.id);

        let root = setup.handles.error(&t1).expect("errored").clone();
        let mid = setup
            .handles
            .error(&t2)
            .expect("skipped: its input was errored");
        let tail = setup
            .handles
            .error(&t3)
            .expect("skipped, two below the root");

        assert!(root.same_root(mid), "the same failure, not a new one");
        assert!(root.same_root(tail), "still the same failure at the tail");
        assert!(
            tail.root()
                .contains("this operation cannot serve its problem"),
            "the tail names the original cause: {tail}"
        );
        assert_eq!((root.depth(), mid.depth(), tail.depth()), (0, 1, 2));
    }

    /// An error is released by the tensor's own `Drop`, like any other handle.
    /// That is what bounds the set: it holds exactly the errored tensors that
    /// are still alive, and needs no cap or eviction of its own.
    #[test]
    fn an_error_is_released_when_its_tensor_is_dropped() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);
        assert!(setup.handles.error(&t1).is_some());

        setup.handles.free(&tensor_ir(t1, TensorStatus::ReadWrite));
        assert!(
            setup.handles.error(&t1).is_none(),
            "the error goes with the tensor"
        );
    }

    /// The stream is not rebuilt around a failure, so what it had queued is
    /// still there and still runs.
    ///
    /// A failure that discarded the segment's operations would leave `global`
    /// and `relative` describing closures that no longer exist, and the next
    /// plan to match would index into the gap
    /// (`OrderedExecution::execute_operations`, index out of bounds).
    #[test]
    fn the_queue_survives_a_failure_intact() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);
        let t2 = TensorId::new(2);

        setup.register_exp(t0, t1);
        setup.streams.register(
            setup.id,
            exp_op(t1, t2),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);
        assert_eq!(setup.num_pending(), 0, "no orphaned bookkeeping survives");

        let t3 = TensorId::new(3);
        let t4 = TensorId::new(4);
        setup.register_exp(t3, t4);
        setup.streams.drain(&mut setup.handles, setup.id);
        assert!(setup.handles.has_handle(&t4), "later work runs normally");
    }

    /// The same on the lazy path, where the operation runs inside the
    /// registration itself — a fire-and-forget `submit` with no caller
    /// blocked on it. Nothing has to be held for a caller that may never
    /// come back, because the error on the tensor is the whole report.
    #[test]
    fn a_failure_while_registering_errors_without_holding_anything() {
        let mut setup = TestSetup::eager();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);
        let t2 = TensorId::new(2);

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        assert!(
            setup.handles.has_handle(&t1),
            "eager fusion executes on registration, not at the drain"
        );

        let registered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            setup.streams.register(
                setup.id,
                exp_op(t1, t2),
                UnfusedOp::new(PanicOp, setup.id),
                &mut setup.handles,
            );
        }));
        assert!(registered.is_ok(), "a registration does not unwind");
        assert!(
            setup.handles.error(&t2).is_some(),
            "the failure is on the tensor it was going to write"
        );
        assert!(
            setup.handles.has_handle(&t1),
            "the input it read is untouched"
        );
    }

    /// A share of an errored tensor carries the error across. Without it the
    /// alias would be a plain missing handle on the receiving stream, and the
    /// thread that reads it would be told the tensor does not exist rather
    /// than why it was never written.
    #[test]
    fn a_share_of_an_errored_tensor_carries_the_error() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);
        let alias = TensorId::new(2);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );

        let shared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            setup
                .streams
                .tag_shared_view(setup.id, t1, alias, &mut setup.handles);
        }));
        assert!(shared.is_ok(), "a share does not unwind into its caller");
        assert!(
            !setup.handles.has_handle(&alias),
            "there is no data to alias"
        );

        let source = setup.handles.error(&t1).expect("the source is errored");
        let aliased = setup.handles.error(&alias).expect("so is the alias");
        assert!(
            source.same_root(aliased),
            "the alias names the same failure, not one of its own"
        );
        assert!(
            !setup.streams.shared_sources.contains(&t1),
            "an unmaterialised source must not be recorded as drained"
        );
    }

    /// A handle registered before the work that fills it is not a written
    /// tensor. In-place fusion registers an output as an alias of its input
    /// while the launch is still being planned, so an error that skipped
    /// tensors "that already have a handle" would leave that output reading
    /// back as a half-written buffer.
    #[test]
    fn an_error_displaces_a_handle_registered_ahead_of_the_work() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(AliasThenPanicOp { out: t1 }, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);

        assert!(
            setup.handles.error(&t1).is_some(),
            "the output holds the error even though it had a handle when the work failed"
        );
        assert!(!setup.handles.has_handle(&t1));
    }

    /// An error must not outlive the tensor carrying it, which means the drop
    /// that releases it has to run. A drop names its tensor as an *input*, so
    /// treating it like any other operation would skip it on the error it is
    /// there to release — and the error would then be held for the life of
    /// the server, for a tensor nobody can even name any more.
    #[test]
    fn a_drop_of_an_errored_tensor_still_releases_it() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);
        assert!(setup.handles.error(&t1).is_some(), "errored");

        // The last `FusionTensor` for t1 goes out of scope.
        let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        setup.streams.register(
            setup.id,
            OperationIr::Drop(tensor_ir(t1, TensorStatus::ReadWrite)),
            UnfusedOp::new(
                DropOp {
                    id: t1,
                    ran: ran.clone(),
                },
                setup.id,
            ),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);

        assert!(
            ran.load(std::sync::atomic::Ordering::Relaxed),
            "the drop must not be skipped on the error it is there to release"
        );
        assert!(
            setup.handles.error(&t1).is_none(),
            "the error went with the tensor"
        );
        assert_eq!(setup.handles.num_handles(), 1, "only t0 remains");
    }

    /// An errored tensor has no producer left to wait for, so the read must
    /// not be sent round a drain that cannot change the answer.
    #[test]
    fn a_read_of_an_errored_tensor_does_not_force_a_drain() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            exp_op(t0, t1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);

        let plan = setup.streams.read_plan(
            setup.id,
            &tensor_ir(t1, TensorStatus::ReadWrite),
            &setup.handles,
        );
        assert!(matches!(plan, ReadPlan::Direct));
    }

    /// A deferred free belongs to a tensor whose `FusionTensor` is already
    /// gone, so no `Drop` op is ever coming for it — only an execution
    /// boundary frees it. The boundary must still be reached when the segment
    /// contained a failure, or the handle is stranded for the life of the
    /// server.
    #[test]
    fn a_failing_segment_still_reaches_its_execution_boundary() {
        let mut setup = TestSetup::new();
        let t0 = TensorId::new(0);
        let t1 = TensorId::new(1);
        let t2 = TensorId::new(2);

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);

        // Cross-thread last use of t0 while a pending op still reads it.
        setup.consuming_read(t0);
        assert!(setup.handles.has_handle(&t0), "deferred, not yet freed");

        setup.streams.register(
            setup.id,
            exp_op(t1, t2),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);

        assert!(
            !setup.handles.has_handle(&t0),
            "the boundary released what only this stream could release"
        );
    }

    /// A backend whose handles live where the operation never reached has to
    /// hear that a drained operation did not run.
    ///
    /// The remote backend reclaims a replayed operation's inputs by
    /// suppressing their client-side `Drop`, because the server already
    /// popped them. An operation that was skipped or torn down was never
    /// replayed, so suppressing it there strands the buffer on the server for
    /// the life of the session.
    #[test]
    fn an_operation_that_did_not_run_reclaims_its_inputs_as_such() {
        UNRUN_FREES.with(|freed| freed.borrow_mut().clear());

        let mut setup = TestSetup::new();
        let (a0, a1, a2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));
        let (b0, b1) = (TensorId::new(10), TensorId::new(11));

        setup.handles.register_handle(a0, TestHandle);
        setup.handles.register_handle(b0, TestHandle);

        // a0 -> a1 fails, so a1 -> a2 behind it is skipped. b0 -> b1 runs.
        setup.streams.register(
            setup.id,
            consume_op(a0, a1),
            UnfusedOp::new(PanicOp, setup.id),
            &mut setup.handles,
        );
        setup.streams.register(
            setup.id,
            consume_op(a1, a2),
            UnfusedOp::new(ProduceOp { out: a2 }, setup.id),
            &mut setup.handles,
        );
        setup.streams.register(
            setup.id,
            consume_op(b0, b1),
            UnfusedOp::new(ProduceOp { out: b1 }, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);

        let unrun = UNRUN_FREES.with(|freed| freed.borrow().clone());
        assert!(
            unrun.contains(&a0),
            "the input of the operation that failed: {unrun:?}"
        );
        assert!(
            unrun.contains(&a1),
            "the input of the operation that skipped: {unrun:?}"
        );
        assert!(
            !unrun.contains(&b0),
            "but not the input of the one that ran: {unrun:?}"
        );
    }

    /// The control for the fused tests below: a block really does compile to
    /// one kernel, and that kernel writes every output in it. Without this,
    /// the failure tests could pass on a block that never ran.
    #[test]
    fn a_fused_kernel_writes_the_whole_blocks_output_set() {
        let mut setup = TestSetup::fused();
        let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        setup.register_exp(t1, t2);

        assert_eq!(setup.num_pending(), 0, "the block closed and ran");
        assert!(setup.handles.has_handle(&t1), "the intermediate is written");
        assert!(setup.handles.has_handle(&t2), "and so is the output");
    }

    /// A fused kernel is one unit of work: it writes every output of every
    /// operation it replaced, so a panic anywhere in it leaves the whole
    /// write set unwritten. Erroring only the last operation's output would
    /// leave the intermediates readable as whatever the allocation held.
    #[test]
    fn a_failing_fused_kernel_errors_the_whole_block() {
        let mut setup = TestSetup::fused();
        let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        setup.streams.register(
            setup.id,
            failing_op(t1, t2),
            UnfusedOp::new(ProduceOp { out: t2 }, setup.id),
            &mut setup.handles,
        );

        let intermediate = setup
            .handles
            .error(&t1)
            .expect("the intermediate the kernel never wrote");
        let output = setup.handles.error(&t2).expect("and its output");
        assert!(
            intermediate.same_root(output),
            "one failure, not one per operation"
        );
        assert!(
            output
                .root()
                .contains("this fused kernel cannot serve its problem"),
            "the error names the kernel that raised it: {output}"
        );
        assert!(!setup.handles.has_handle(&t1));
        assert!(!setup.handles.has_handle(&t2));
        assert!(setup.handles.has_handle(&t0), "its input is untouched");
    }

    /// One errored input anywhere in a fused block stops the whole thing: the
    /// kernel reads every input of every operation it replaced, so there is no
    /// part of it that could still run. The block's outputs report the failure
    /// that started it rather than one of their own.
    #[test]
    fn a_fused_block_with_an_errored_input_never_runs() {
        let mut setup = TestSetup::fused();
        let t0 = TensorId::new(0);
        let (t1, t2, t3) = (TensorId::new(1), TensorId::new(2), TensorId::new(3));

        setup.handles.register_handle(t0, TestHandle);
        setup.streams.register(
            setup.id,
            failing_op(t0, t1),
            UnfusedOp::new(ProduceOp { out: t1 }, setup.id),
            &mut setup.handles,
        );
        setup.streams.drain(&mut setup.handles, setup.id);
        let root = setup.handles.error(&t1).expect("errored").clone();

        // A two-operation block reading what the failure never wrote.
        setup.register_exp(t1, t2);
        setup.register_exp(t2, t3);

        let mid = setup.handles.error(&t2).expect("the block was skipped");
        let tail = setup.handles.error(&t3).expect("all of it");
        assert!(root.same_root(mid), "the same failure, not a new one");
        assert!(root.same_root(tail));
        assert!(
            !setup.handles.has_handle(&t2) && !setup.handles.has_handle(&t3),
            "a skipped kernel writes nothing"
        );
        assert_eq!(
            (root.depth(), mid.depth(), tail.depth()),
            (0, 1, 1),
            "the block is one hop from the failure, not one hop per operation"
        );
    }

    /// The backstop in `run_strategy`. Each unit of work catches its own
    /// panic, so nothing should unwind out of the strategy walk — but a plan
    /// that does not fit the stream it matched can panic in the walk itself,
    /// and that frame is the only one still holding the queue's lists. They
    /// have to come back, or the next plan to match indexes into the gap.
    #[test]
    fn a_panic_escaping_the_strategy_leaves_the_queue_usable() {
        use crate::search::BlockOptimization;
        use crate::stream::store::ExecutionStrategy;

        let mut setup = TestSetup::new();
        let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));

        setup.handles.register_handle(t0, TestHandle);
        setup.register_exp(t0, t1);
        setup.register_exp(t1, t2);

        // Two operations queued, a plan naming a third: the first runs, then
        // the walk indexes past the end of the segment.
        let ordering = vec![0, 2];
        let strategy = ExecutionStrategy::Operations {
            ordering: std::sync::Arc::new(ordering.clone()),
        };
        let stream = setup
            .streams
            .streams
            .get_mut(&setup.id)
            .expect("the stream exists");
        let escaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stream.queue.execute_unfused(
                BlockOptimization::new(strategy, ordering),
                &mut setup.handles,
                setup.id,
            );
        }));

        assert!(escaped.is_ok(), "the panic does not reach the caller");
        assert!(
            setup.handles.has_handle(&t1),
            "an output that was written is left alone: nothing says the \
             failure came from the operation that wrote it"
        );
        assert!(
            setup.handles.error(&t2).is_some(),
            "one that was not carries the failure"
        );

        // The lists came back, and in step with each other. Dropping them
        // mid-unwind is what left `global` and `relative` describing closures
        // that were gone, for the next plan to match and index into.
        let queue = &setup
            .streams
            .streams
            .get(&setup.id)
            .expect("the stream exists")
            .queue;
        assert_eq!(
            (queue.global.len(), queue.operations.len()),
            (0, 0),
            "the consumed segment left no orphaned bookkeeping"
        );
    }
}
