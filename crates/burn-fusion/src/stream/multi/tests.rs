//! Cross-thread last-use (consuming read or foreign `Drop`) of a materialized
//! tensor. The core property is determinism: the composition must depend on
//! the home thread's op sequence alone, never on when the foreign message
//! lands (the `*_timing_*` tests). The rest pin the free-timing contract:
//! deferred only while pending ops still reference the tensor, released at
//! the next boundary, immediate otherwise.

use super::*;
use crate::{
    FuserProperties, FuserStatus, NumOperations, OperationFuser, OperationRan, Optimization,
    UnfusedOp,
    stream::{Context, Operation, OrderedExecution},
};
use burn_backend::{DType, DeviceId, DeviceOps, DeviceSettings, Shape};
use burn_ir::{FloatOperationIr, TensorError, TensorIr, TensorStatus, UnaryOpIr};
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
    /// Operations the kernel refuses to serve and runs unfused instead,
    /// as indices within the block — what a real optimization does when
    /// part of what it replaced needs the fallback.
    fallback: Vec<usize>,
    /// Global ids the kernel claims before falling back, standing in for a
    /// fused step that failed to write them part way through the block.
    claims: Vec<TensorId>,
    /// Global ids the kernel writes, for a block built by hand rather than
    /// by a fuser — where there are no relative ids to resolve through.
    writes: Vec<TensorId>,
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

        for id in &self.claims {
            context.handles.set_error(
                *id,
                TensorError::panicked("the fused part could not write it"),
            );
        }

        for index in &self.fallback {
            _execution
                .operation_within_optimization(*index)
                .execute(&mut context.handles);
        }

        for id in &self.writes {
            context.handles.register_handle(*id, TestHandle);
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
            fallback: Vec::new(),
            claims: Vec::new(),
            writes: Vec::new(),
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
    fn execute(
        &self,
        handles: &mut HandleContainer<TestHandle>,
    ) -> Result<(), burn_backend::ExecutionError> {
        handles.register_handle(self.out, TestHandle);

        Ok(())
    }
}

/// Declines to run and says why, the way a backend that can report does.
/// Its write set is claimed exactly as a panicking one's is; the difference
/// is only that the claim carries a typed error rather than a message.
#[derive(Debug)]
struct ReportOp;

impl Operation<TestRuntime> for ReportOp {
    fn execute(
        &self,
        _handles: &mut HandleContainer<TestHandle>,
    ) -> Result<(), burn_backend::ExecutionError> {
        Err(burn_backend::ExecutionError::generic(
            "this operation declined to run",
        ))
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
    fn execute(
        &self,
        handles: &mut HandleContainer<TestHandle>,
    ) -> Result<(), burn_backend::ExecutionError> {
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
    fn execute(
        &self,
        handles: &mut HandleContainer<TestHandle>,
    ) -> Result<(), burn_backend::ExecutionError> {
        self.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        handles.remove_handle(self.id);

        Ok(())
    }
}

/// Panics when executed, the way a pinned kernel that cannot serve its
/// problem does in the benchmark sweeps downstream.
#[derive(Debug)]
struct PanicOp;

impl Operation<TestRuntime> for PanicOp {
    fn execute(
        &self,
        _handles: &mut HandleContainer<TestHandle>,
    ) -> Result<(), burn_backend::ExecutionError> {
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

    // Reading it is where a caller finally hears about it — as an error
    // it can handle, not an unwind.
    let read = setup
        .handles
        .take_error(&tensor_ir(t1, TensorStatus::ReadWrite))
        .expect("the read must not hand back untouched memory");
    assert!(
        read.root()
            .contains("this operation cannot serve its problem"),
        "the read names the root cause: {read}"
    );

    // And the read consumed the tensor, so it released the failure with
    // it: the claim lives exactly as long as the tensor carrying it.
    assert!(
        !setup.handles.has_errors(),
        "the failure is released with the tensor the read consumed"
    );
}

/// A read that does not consume the tensor leaves the failure in place:
/// the tensor is still alive, and the next read of it has to report the
/// same cause rather than a bare missing handle.
#[test]
fn a_read_only_read_leaves_the_failure_for_the_next_one() {
    let mut setup = TestSetup::new();
    let (t0, t1) = (TensorId::new(0), TensorId::new(1));
    setup.handles.register_handle(t0, TestHandle);
    setup.streams.register(
        setup.id,
        exp_op(t0, t1),
        UnfusedOp::new(PanicOp, setup.id),
        &mut setup.handles,
    );
    setup.streams.drain(&mut setup.handles, setup.id);

    let first = setup
        .handles
        .take_error(&tensor_ir(t1, TensorStatus::ReadOnly));
    let second = setup
        .handles
        .take_error(&tensor_ir(t1, TensorStatus::ReadOnly));

    assert!(first.is_some() && second.is_some(), "both reads report it");
    assert!(
        first.zip(second).is_some_and(|(a, b)| a.same_root(&b)),
        "and report the same failure, not a fresh one"
    );
    assert!(setup.handles.has_errors(), "the tensor still carries it");
}

/// A plan names operation indices in the stream it was cached from, and is
/// matched against one it did not run on, so the indices are a claim rather
/// than a fact. One that does not fit costs fusion, not the work: the
/// operations are all still here, so they run in submission order, which is
/// always a legal order.
#[test]
fn a_plan_that_does_not_fit_runs_its_operations_unfused() {
    use crate::search::BlockOptimization;
    use crate::stream::store::ExecutionStrategy;

    let mut setup = TestSetup::new();
    let (t0, t1) = (TensorId::new(0), TensorId::new(1));
    setup.handles.register_handle(t0, TestHandle);
    setup.register_exp(t0, t1);

    let stream = setup
        .streams
        .streams
        .get_mut(&setup.id)
        .expect("the registration created the stream");
    assert_eq!(stream.queue.global.len(), 1, "one operation queued");

    // A plan naming three operations against a segment holding one.
    let ordering = vec![0, 1, 2];
    let optimization = BlockOptimization::new(
        ExecutionStrategy::Optimization {
            opt: TestOptimization {
                len: ordering.len(),
                outputs: Vec::new(),
                panics: false,
                fallback: Vec::new(),
                claims: Vec::new(),
                writes: Vec::new(),
            },
            ordering: std::sync::Arc::new(ordering.clone()),
            score: 0,
        },
        ordering,
    );

    stream
        .queue
        .execute_unfused(optimization, &mut setup.handles, setup.id);

    assert!(
        stream.queue.global.is_empty() && stream.queue.operations.is_empty(),
        "the segment was consumed, so the next pass cannot re-select it"
    );
    assert!(
        setup.handles.has_handle(&t1),
        "and the operation ran: an unfitting plan must not cost the work"
    );
    assert!(
        !setup.handles.has_errors(),
        "nothing failed, so nothing is claimed"
    );
}

/// A fallback that skips was never replayed server-side, so it has to reach
/// `did_not_run` like any other work that did not run — a runtime whose
/// handles live on a server strands the buffer otherwise. It cannot hold a
/// reference to the execution it came from, so it records through a shared
/// one, and this is what checks that the two meet again.
#[test]
fn a_skipped_fallback_is_recorded_as_not_run() {
    use crate::search::BlockOptimization;
    use crate::stream::store::ExecutionStrategy;

    UNRUN_FREES.with(|freed| freed.borrow_mut().clear());

    let mut setup = TestSetup::new();
    let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));
    setup.handles.register_handle(t0, TestHandle);
    setup.streams.register(
        setup.id,
        consume_op(t0, t1),
        UnfusedOp::new(ProduceOp { out: t1 }, setup.id),
        &mut setup.handles,
    );
    setup.streams.register(
        setup.id,
        consume_op(t1, t2),
        UnfusedOp::new(ProduceOp { out: t2 }, setup.id),
        &mut setup.handles,
    );

    // The kernel fails to write `t1`, then falls back for the operation
    // reading it — which must skip, and must say that it did.
    let ordering = vec![0, 1];
    let optimization = BlockOptimization::new(
        ExecutionStrategy::Optimization {
            opt: TestOptimization {
                len: ordering.len(),
                outputs: Vec::new(),
                panics: false,
                fallback: vec![1],
                claims: vec![t1],
                writes: Vec::new(),
            },
            ordering: std::sync::Arc::new(ordering.clone()),
            score: 0,
        },
        ordering,
    );

    let stream = setup.streams.streams.get_mut(&setup.id).expect("queued");
    stream
        .queue
        .execute_unfused(optimization, &mut setup.handles, setup.id);

    assert!(
        setup.handles.error(&t2).is_some(),
        "the fallback skipped, so its output is claimed"
    );

    // `t1` is the skipped operation's last-use input. Reclaiming it as
    // `OperationRan::No` is what tells a runtime holding it elsewhere that
    // nothing replayed the operation that would have freed it.
    let unrun = UNRUN_FREES.with(|freed| freed.borrow().clone());
    assert!(
        unrun.contains(&t1),
        "the skipped fallback must be recorded as not run, got {unrun:?}"
    );
}

/// The recovery property autotune rests on: a candidate that fails claims
/// the output it did not write, and the candidate that works writes it and
/// clears the claim. Nothing downstream is skipped, and the read succeeds.
#[test]
fn writing_a_claimed_tensor_recovers_it() {
    let mut setup = TestSetup::new();
    let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));
    setup.handles.register_handle(t0, TestHandle);

    // The failing candidate: it claims `t1`, which it never wrote.
    setup.streams.register(
        setup.id,
        exp_op(t0, t1),
        UnfusedOp::new(PanicOp, setup.id),
        &mut setup.handles,
    );
    setup.streams.drain(&mut setup.handles, setup.id);
    assert!(setup.handles.error(&t1).is_some(), "claimed by the failure");

    // The candidate that works, writing the same output.
    setup.handles.register_handle(t1, TestHandle);

    assert!(setup.handles.error(&t1).is_none(), "the claim is cleared");
    assert!(setup.handles.has_handle(&t1), "and the bytes are there");
    assert!(
        !setup.handles.has_errors(),
        "so the container is clean again"
    );

    // Work reading the recovered tensor runs, rather than being skipped.
    setup.handles.register_handle(t0, TestHandle);
    setup.register_exp(t1, t2);
    setup.streams.drain(&mut setup.handles, setup.id);
    assert!(setup.handles.has_handle(&t2), "downstream work ran");
    assert!(setup.handles.error(&t2).is_none(), "and holds no error");
}

/// Unfused work inside a fused block obeys the same rule as unfused work
/// outside one. An optimization that cannot serve part of what it replaced
/// runs those operations directly, and that fallback must not be the one
/// place a claimed tensor still reaches a kernel: it skips, and its output
/// carries the failure that stopped it.
#[test]
fn a_fallback_does_not_run_on_a_claimed_input() {
    use crate::search::BlockOptimization;
    use crate::stream::store::ExecutionStrategy;

    let mut setup = TestSetup::new();
    let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));
    setup.handles.register_handle(t0, TestHandle);
    setup.register_exp(t0, t1);
    setup.register_exp(t1, t2);

    let stream = setup
        .streams
        .streams
        .get_mut(&setup.id)
        .expect("the registration created the stream");

    // The kernel fails to write `t1` part way through, then falls back for
    // the operation that reads it. Nothing claimed `t1` when the block was
    // entered, so the block-level check let the whole thing through.
    let ordering = vec![0, 1];
    let optimization = BlockOptimization::new(
        ExecutionStrategy::Optimization {
            opt: TestOptimization {
                len: ordering.len(),
                outputs: Vec::new(),
                panics: false,
                fallback: vec![1],
                claims: vec![t1],
                writes: Vec::new(),
            },
            ordering: std::sync::Arc::new(ordering.clone()),
            score: 0,
        },
        ordering,
    );

    stream
        .queue
        .execute_unfused(optimization, &mut setup.handles, setup.id);

    assert!(
        !setup.handles.has_handle(&t2),
        "the fallback must not run on a tensor nothing wrote"
    );
    let claimed = setup.handles.error(&t2).expect("its output is claimed");
    assert_eq!(
        claimed.root(),
        "the fused part could not write it",
        "and names the failure that stopped it"
    );
    assert_eq!(claimed.depth(), 1, "one hop below that failure");
}

/// A timing harness for the execution path, not an assertion.
///
/// The claim bookkeeping runs per operation on the success path, so a
/// change to what a scope does on the way in is paid by every operation a
/// program executes, not only by the ones that fail. Run it before and
/// after such a change:
///
/// ```text
/// cargo test -p burn-fusion --release execution_path_throughput -- --ignored --nocapture
/// ```
#[test]
#[ignore = "timing harness; run explicitly"]
fn execution_path_throughput() {
    const CHAIN: u64 = 32;
    const ROUNDS: usize = 20_000;

    let mut setup = TestSetup::new();
    setup.handles.register_handle(TensorId::new(0), TestHandle);

    let start = std::time::Instant::now();
    for _ in 0..ROUNDS {
        for id in 0..CHAIN {
            setup.register_exp(TensorId::new(id), TensorId::new(id + 1));
        }
        setup.streams.drain(&mut setup.handles, setup.id);
        // Keep the container small: the next round rewrites the same ids.
        for id in 1..=CHAIN {
            setup.handles.remove_handle(TensorId::new(id));
        }
        setup.handles.register_handle(TensorId::new(0), TestHandle);
    }
    let elapsed = start.elapsed();

    let operations = ROUNDS as u64 * CHAIN;
    println!(
        "{operations} operations in {elapsed:?} — {:.0} ns/op",
        elapsed.as_nanos() as f64 / operations as f64
    );
}

/// The one oracle this area answers to, checked over random interleavings.
///
/// > A tensor reads back if and only if the work that was going to write it
/// > ran and succeeded.
///
/// The harness drives the same machinery a backend drives — a queue of
/// operations, executed under the strategy the planner would have chosen —
/// over random sequences of register, fail, report, drain and recover.
///
/// The unfused path, where a claim is per operation and the model can be
/// written independently of the implementation. A fused block claims all of
/// its outputs together, and modelling that means modelling where the fuser
/// puts block boundaries — a model that is a copy of the implementation
/// checks nothing. That granularity is pinned by targeted tests instead.
///
/// The model keeps its own answer per tensor — one enum, deliberately
/// nothing like the propagation it checks — and is compared after every
/// drain. Two invariants ride along:
///
/// - the container's claim count never drifts from the model's, which is the
///   bound the whole design rests on: a claim lives exactly as long as the
///   tensor carrying it;
/// - tensors claimed by one failure all report the same root, however far
///   apart in the chain they are.
#[test]
fn a_tensor_reads_back_only_if_the_work_writing_it_succeeded() {
    /// What the model believes about one tensor.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Truth {
        /// Nothing has written it and nothing claims it.
        Absent,
        /// The work that writes it ran and succeeded.
        Written,
        /// A failure claims it. Carries which failure, so the roots the
        /// container reports can be checked against one another.
        Claimed(u32),
    }

    /// A queued operation, and what the model expects of it.
    struct Queued {
        input: TensorId,
        out: TensorId,
        /// Whether the operation refuses to run when it is reached.
        fails: bool,
    }

    // A tiny LCG, so a failing seed reproduces exactly with no dev-dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self, bound: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as usize) % bound.max(1)
        }
    }

    const TENSORS: u64 = 6;
    const SEEDS: u64 = 40;
    const STEPS: usize = 60;

    for seed in 0..SEEDS {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
        let mut setup = TestSetup::new();
        let mut truth = vec![Truth::Absent; TENSORS as usize];
        let mut failure = 0u32;
        let mut queued: Vec<Queued> = Vec::new();

        // Seed one tensor with data, so there is something to read from.
        setup.handles.register_handle(TensorId::new(0), TestHandle);
        truth[0] = Truth::Written;

        for _ in 0..STEPS {
            match rng.next(4) {
                // Queue an operation reading a tensor that holds something.
                0 | 1 => {
                    let readable: Vec<usize> = (0..TENSORS as usize)
                        .filter(|id| truth[*id] != Truth::Absent)
                        .collect();
                    if readable.is_empty() {
                        continue;
                    }
                    let input = readable[rng.next(readable.len())];
                    let out = rng.next(TENSORS as usize);
                    // An output that is still an input of something queued
                    // would make the model depend on ordering it does not
                    // track; keep each queued output distinct.
                    if queued.iter().any(|q| q.out.value() as usize == out) || out == input {
                        continue;
                    }
                    let fails = rng.next(4) == 0;

                    let input = TensorId::new(input as u64);
                    let out = TensorId::new(out as u64);
                    let ir = exp_op(input, out);
                    // A failure reports or raises; the claim is the same
                    // either way, which is the point.
                    let op = match (fails, rng.next(2) == 0) {
                        (true, true) => UnfusedOp::new(PanicOp, setup.id),
                        (true, false) => UnfusedOp::new(ReportOp, setup.id),
                        (false, _) => UnfusedOp::new(ProduceOp { out }, setup.id),
                    };
                    setup.streams.register(setup.id, ir, op, &mut setup.handles);
                    queued.push(Queued { input, out, fails });
                }
                // Drain through the processor, the way a stream really
                // drains. Driving `execute_unfused` directly would let the
                // queue and the planner's stored state fall out of step,
                // which is a state production never reaches.
                2 => {
                    if queued.is_empty() {
                        continue;
                    }

                    for q in &queued {
                        truth[q.out.value() as usize] = match truth[q.input.value() as usize] {
                            Truth::Claimed(root) => Truth::Claimed(root),
                            _ if q.fails => {
                                failure += 1;
                                Truth::Claimed(failure)
                            }
                            _ => Truth::Written,
                        };
                    }

                    setup.streams.drain(&mut setup.handles, setup.id);
                    queued.clear();
                }
                // Recover a claimed tensor by writing it.
                _ => {
                    let claimed: Vec<usize> = (0..TENSORS as usize)
                        .filter(|id| matches!(truth[*id], Truth::Claimed(_)))
                        .filter(|id| !queued.iter().any(|q| q.out.value() as usize == *id))
                        .collect();
                    if claimed.is_empty() {
                        continue;
                    }
                    let id = claimed[rng.next(claimed.len())];
                    setup
                        .handles
                        .register_handle(TensorId::new(id as u64), TestHandle);
                    truth[id] = Truth::Written;
                }
            }

            // Nothing is checked while operations are still queued: the
            // model describes what a drain will have done, not what the
            // container holds part way there.
            if !queued.is_empty() {
                continue;
            }

            let mut roots: HashMap<u32, TensorError> = HashMap::new();
            let mut claims = 0;

            for (id, expected) in truth.iter().enumerate() {
                let tensor = TensorId::new(id as u64);
                match *expected {
                    Truth::Absent => {
                        assert!(
                            !setup.handles.has_handle(&tensor)
                                && setup.handles.error(&tensor).is_none(),
                            "seed {seed}: {tensor:?} should hold nothing"
                        );
                    }
                    Truth::Written => {
                        assert!(
                            setup.handles.has_handle(&tensor),
                            "seed {seed}: {tensor:?} was written and must read back"
                        );
                        assert!(
                            setup.handles.error(&tensor).is_none(),
                            "seed {seed}: {tensor:?} was written and must not be claimed"
                        );
                    }
                    Truth::Claimed(root) => {
                        claims += 1;
                        let claim = setup.handles.error(&tensor).unwrap_or_else(|| {
                            panic!("seed {seed}: {tensor:?} was never written and must be claimed")
                        });
                        assert!(
                            !setup.handles.has_handle(&tensor),
                            "seed {seed}: {tensor:?} is claimed, so it has no data"
                        );
                        match roots.get(&root) {
                            Some(first) => assert!(
                                first.same_root(claim),
                                "seed {seed}: {tensor:?} should share one failure's root"
                            ),
                            None => {
                                roots.insert(root, claim.clone());
                            }
                        }
                    }
                }
            }

            assert_eq!(
                setup.handles.has_errors(),
                claims > 0,
                "seed {seed}: the claim count drifted from the map"
            );
        }
    }
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

/// The same check covers an unfused plan whose ordering names an operation
/// the segment does not have. Nothing indexes past the end part way through
/// the walk, where the panic would be raised outside every scope with
/// nothing able to say what it left unwritten.
#[test]
fn an_out_of_range_ordering_never_reaches_the_walk() {
    use crate::search::BlockOptimization;
    use crate::stream::store::ExecutionStrategy;

    let mut setup = TestSetup::new();
    let (t0, t1, t2) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));

    setup.handles.register_handle(t0, TestHandle);
    setup.register_exp(t0, t1);
    setup.register_exp(t1, t2);

    // Two operations queued, a plan naming a third.
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

    assert!(escaped.is_ok(), "nothing panics");
    assert!(
        setup.handles.has_handle(&t1) && setup.handles.has_handle(&t2),
        "both operations ran, in submission order"
    );
    assert!(!setup.handles.has_errors(), "so nothing is claimed");

    // The lists came back, and in step with each other.
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
