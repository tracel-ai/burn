use std::sync::Arc;

use burn_ir::{HandleContainer, OperationIr, TensorError};

use crate::{FusionRuntime, NumOperations, Optimization, UnfusedOp, stream::Context};

/// The failure claiming any tensor `op` reads — the check a unit of work
/// makes before it runs.
///
/// Work whose input was never written must not run: those bytes are whatever
/// the allocation happened to hold, and computing on them turns a failure
/// that named one tensor into a wrong answer that names none. The outputs
/// take the same claim instead, so a read below the skip still reports the
/// failure that started it.
///
/// `inputs()` rather than `nodes()`: this runs before every operation on the
/// hot path, and `nodes()` collects into a fresh `Vec` to chain the two.
pub(crate) fn input_failure<'a, H>(
    op: &OperationIr,
    handles: &'a HandleContainer<H>,
) -> Option<&'a TensorError>
where
    H: Clone,
{
    // Nothing is claimed, so nothing can be found — and asking anyway would
    // cost a boxed iterator per operation for an answer that is always
    // `None`. This runs before every operation, so the check has to be free
    // while nothing has failed.
    if !handles.has_claims() {
        return None;
    }

    // A drop names its tensor as an input, but it does not read it — it is
    // what releases it, and releasing is how a claim stops being held. Skip
    // it and the claim outlives every tensor that could report it, for the
    // life of the server: the bound this whole design rests on is that a
    // claim lives exactly as long as the tensor carrying it.
    if let OperationIr::Drop(_) = op {
        return None;
    }

    op.inputs().find_map(|node| handles.error(&node.id))
}

/// Claim every tensor `op` was going to write, so a read of one reports
/// `error` instead of handing back bytes nothing wrote.
///
/// Unconditional, because these are exactly the tensors this operation was
/// responsible for: an in-place output has a handle registered while the
/// launch is still being planned, so finding one there says nothing about
/// whether the kernel that fills it ever ran.
pub(crate) fn claim_outputs<H>(
    op: &OperationIr,
    handles: &mut HandleContainer<H>,
    error: &TensorError,
) where
    H: Clone,
{
    for node in op.outputs() {
        handles.claim(node.id, error.clone());
    }
}

/// The message inside a caught panic payload. Covers what `panic!` produces:
/// `&'static str` and `String`.
pub(crate) fn panic_message(panic: &(dyn core::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

/// What a finished [`OrderedExecution`] hands back to the queue.
pub(crate) struct Executed<R: FusionRuntime> {
    /// The operations it did not consume, to go back on the queue.
    pub(crate) operations: Vec<UnfusedOp<R>>,
    /// The segment's IR, likewise.
    pub(crate) ir: Vec<OperationIr>,
    /// How many operations it consumed, run or claimed.
    pub(crate) num_executed: usize,
    /// The first panic raised, kept only so the caller can log it — every
    /// failure's report is the claim it left on the tensors.
    pub(crate) failed: Option<Box<dyn core::any::Any + Send>>,
}

/// Manage the execution of potentially multiple optimizations and operations out of order.
pub struct OrderedExecution<R: FusionRuntime> {
    operations: Vec<UnfusedOp<R>>,
    /// The segment's operation IR, parallel to `operations`. What each unit
    /// of work reads and writes, which is what decides whether it may run and
    /// what it claims when it does not. Moved in and out with `operations`
    /// rather than copied, so carrying it costs nothing per segment.
    ir: Vec<OperationIr>,
    num_executed: usize,
    ordering: Option<Arc<Vec<usize>>>,
    /// The first panic a unit of this execution raised, kept only so the
    /// caller can log it. Every failure's report is the claim it left on the
    /// tensors, not this.
    failed: Option<Box<dyn core::any::Any + Send>>,
}

impl<R: FusionRuntime> OrderedExecution<R> {
    /// Returns the operation that can be executed without impacting the state of the execution.
    ///
    /// This is useful to implement fallback for optimizations.
    #[allow(clippy::borrowed_box)]
    pub fn operation_within_optimization(&self, index: usize) -> UnfusedOp<R> {
        match &self.ordering {
            Some(val) => {
                let index = val[index];
                self.operations[index].clone()
            }
            None => panic!("No ordering provided"),
        }
    }

    pub(crate) fn new(operations: Vec<UnfusedOp<R>>, ir: Vec<OperationIr>) -> Self {
        Self {
            operations,
            ir,
            num_executed: 0,
            ordering: None,
            failed: None,
        }
    }

    pub(crate) fn finish(mut self) -> Executed<R> {
        // `min`: the count is claimed before the work runs (see
        // `execute_optimization`), so a strategy torn down by a panic can
        // leave it describing operations a shorter list never held.
        let num_executed = self.num_executed.min(self.operations.len());
        self.operations.drain(0..num_executed);

        Executed {
            operations: self.operations,
            ir: self.ir,
            num_executed,
            failed: self.failed,
        }
    }

    pub(crate) fn execute_optimization(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<R::FusionHandle>,
        ordering: Arc<Vec<usize>>,
    ) {
        if ordering.len() > self.operations.len() {
            panic!(
                "Ordering is bigger than operations: ordering len {}, operations len {}, \
                 num_executed {}, optimization len {}, ordering {:?}",
                ordering.len(),
                self.operations.len(),
                self.num_executed,
                optimization.len(),
                ordering,
            );
        }
        self.ordering = Some(ordering);
        let num_drained = optimization.len();
        // Counted before the call rather than after, so an unwind out of
        // `execute` still leaves these operations consumed. Counting after
        // would put them back on the queue for the next segment to retry,
        // re-running work whose inputs the torn-down execution already took —
        // and their outputs are claimed by the failure instead, which is what
        // a read of one of them has to report.
        self.num_executed += num_drained;

        // A fused kernel is one unit of work: it reads every input of every
        // operation it replaced and writes every output, so one claimed input
        // anywhere in it stops the whole thing, and a panic anywhere in it
        // leaves the whole write set unwritten. Either way the claim lands on
        // all of them together.
        let ordering = self.ordering.clone().expect("just set");

        let skip = ordering
            .iter()
            .filter_map(|id| self.ir.get(*id))
            .find_map(|op| input_failure(op, &context.handles))
            .map(TensorError::propagated);

        if let Some(error) = skip {
            self.claim(&ordering, &mut context.handles, &error);
            return;
        }

        let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            optimization.execute(context, self)
        }));

        if let Err(panic) = executed {
            let error = TensorError::new(panic_message(panic.as_ref()));
            self.claim(&ordering, &mut context.handles, &error);
            self.failed.get_or_insert(panic);
        }
    }

    /// Claim the write sets of every operation at `ordering`.
    fn claim(
        &self,
        ordering: &[usize],
        handles: &mut HandleContainer<R::FusionHandle>,
        error: &TensorError,
    ) {
        for op in ordering.iter().filter_map(|id| self.ir.get(*id)) {
            claim_outputs(op, handles, error);
        }
    }

    pub(crate) fn execute_operations(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        ordering: &[usize],
    ) {
        self.num_executed += ordering.len();

        for id in ordering {
            // A skip: an input this operation needs was never written, so it
            // does not run and its outputs take the same claim — naming the
            // failure that started it rather than one of their own.
            let skip = self
                .ir
                .get(*id)
                .and_then(|ir| input_failure(ir, handles))
                .map(TensorError::propagated);

            if let Some(error) = skip {
                self.claim(&[*id], handles, &error);
                continue;
            }

            // Caught per operation, not per segment: a segment is just what
            // happened to be queued together, so a failure in one operation
            // says nothing about the next one unless they share a tensor — and
            // if they do, the next one skips on the claim its input now
            // carries. Stopping the loop instead would make an unrelated
            // operation's outcome depend on queue order.
            let op = &self.operations[*id];
            let executed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op.execute(handles)));

            if let Err(panic) = executed {
                self.claim(
                    &[*id],
                    handles,
                    &TensorError::new(panic_message(panic.as_ref())),
                );
                self.failed.get_or_insert(panic);
            }
        }
    }
}
