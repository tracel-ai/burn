use std::sync::Arc;

use burn_ir::{HandleContainer, OperationIr, TensorError};

use super::{input_error, panic_message, set_output_errors};

use crate::{FusionRuntime, NumOperations, Optimization, UnfusedOp, stream::Context};

/// What a finished [`OrderedExecution`] hands back to the queue.
pub(crate) struct Executed<R: FusionRuntime> {
    /// The operations it did not consume, to go back on the queue.
    pub(crate) operations: Vec<UnfusedOp<R>>,
    /// The segment's IR, likewise.
    pub(crate) ir: Vec<OperationIr>,
    /// How many operations it consumed, run or errored.
    pub(crate) num_executed: usize,
    /// Which of those consumed operations never ran — skipped on an errored
    /// input, or torn down by a panic. Indices into `ir`. Empty while nothing
    /// has failed, so carrying it allocates nothing on the common path.
    pub(crate) did_not_run: Vec<usize>,
    /// The first panic raised, kept only so the caller can log it — every
    /// failure's report is the error it left on the tensors.
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
    /// Which consumed operations never ran. See
    /// [`Executed::did_not_run`](Executed#structfield.did_not_run).
    did_not_run: Vec<usize>,
    ordering: Option<Arc<Vec<usize>>>,
    /// The first panic a unit of this execution raised, kept only so the
    /// caller can log it. Every failure's report is the error it left on the
    /// tensors, not this.
    failed: Option<Box<dyn core::any::Any + Send>>,
}

/// One operation of an optimization's block, runnable on its own.
///
/// A fallback is unfused work in the middle of a fused block: an optimization
/// that cannot serve part of what it replaced runs those operations directly
/// instead. So it carries the operation's IR alongside the operation, because
/// running it has to apply the same rule the unfused path applies — an
/// operation whose input a failure claims does not run, and its outputs take
/// that failure. Without it the fallback would be the one place left where a
/// claimed tensor still reaches a kernel.
pub struct FallbackOp<R: FusionRuntime> {
    operation: UnfusedOp<R>,
    ir: OperationIr,
}

impl<R: FusionRuntime> Clone for FallbackOp<R> {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
            ir: self.ir.clone(),
        }
    }
}

impl<R: FusionRuntime> FallbackOp<R> {
    /// Run it — unless a failure claims one of its inputs, in which case its
    /// outputs take that failure, one hop further down, and no kernel runs.
    ///
    /// A panic out of the operation is left to the caller: a fallback runs
    /// inside the optimization's own `catch_unwind`, which claims the whole
    /// block's write set, and that is the honest report for a fused unit that
    /// stopped part way.
    pub fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        if let Some(error) = input_error(&self.ir, handles).map(TensorError::propagated) {
            set_output_errors(&self.ir, handles, &error);
            return;
        }

        self.operation.execute(handles);
    }
}

impl<R: FusionRuntime> OrderedExecution<R> {
    /// Returns the operation that can be executed without impacting the state of the execution.
    ///
    /// This is useful to implement fallback for optimizations.
    #[allow(clippy::borrowed_box)]
    pub fn operation_within_optimization(&self, index: usize) -> FallbackOp<R> {
        match &self.ordering {
            Some(val) => {
                let index = val[index];
                FallbackOp {
                    operation: self.operations[index].clone(),
                    ir: self.ir[index].clone(),
                }
            }
            None => panic!("No ordering provided"),
        }
    }

    pub(crate) fn new(operations: Vec<UnfusedOp<R>>, ir: Vec<OperationIr>) -> Self {
        Self {
            operations,
            ir,
            num_executed: 0,
            did_not_run: Vec::new(),
            ordering: None,
            failed: None,
        }
    }

    pub(crate) fn finish(mut self) -> Executed<R> {
        // `min`: the count is taken before the work runs (see
        // `execute_optimization`), so a strategy torn down by a panic can
        // leave it describing operations a shorter list never held.
        let num_executed = self.num_executed.min(self.operations.len());
        self.operations.drain(0..num_executed);

        Executed {
            operations: self.operations,
            ir: self.ir,
            num_executed,
            did_not_run: self.did_not_run,
            failed: self.failed,
        }
    }

    /// Guarantee forward progress after a panic escaped the strategy walk.
    ///
    /// Every unit of work counts its operations *before* it runs, so an
    /// escape normally leaves at least one consumed and the queue shrinks. A
    /// panic raised before the first count — one of the strategy walk's own
    /// guards — consumes nothing, and an unchanged queue is not a safe place
    /// to stop: the policy re-plans it identically, re-selects the same
    /// strategy and raises the same panic, without end.
    ///
    /// So the block the strategy was going to run is consumed as one unit.
    /// `planned` is that block's length; nothing in it ran, so the claim on
    /// its write set is honest, and the caller reports it exactly as any
    /// other failure.
    pub(crate) fn consume_stalled(&mut self, planned: usize) {
        if self.num_executed > 0 {
            return;
        }

        self.num_executed = planned.min(self.operations.len());
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
        self.ordering = Some(ordering.clone());
        let num_drained = optimization.len();
        // Counted before the call rather than after, so an unwind out of
        // `execute` still leaves these operations consumed. Counting after
        // would put them back on the queue for the next segment to retry,
        // re-running work whose inputs the torn-down execution already took —
        // and their outputs carry the failure instead, which is what a read
        // of one of them has to report.
        self.num_executed += num_drained;

        // A fused kernel is one unit of work: it reads every input of every
        // operation it replaced and writes every output, so one errored input
        // anywhere in it stops the whole thing, and a panic anywhere in it
        // leaves the whole write set unwritten. Either way the error lands on
        // all of them together.
        let skip = ordering
            .iter()
            .map(|id| &self.ir[*id])
            .find_map(|op| input_error(op, &context.handles))
            .map(TensorError::propagated);

        if let Some(error) = skip {
            self.set_errors(&ordering, &mut context.handles, &error);
            self.did_not_run.extend(ordering.iter().copied());
            return;
        }

        let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            optimization.execute(context, self)
        }));

        if let Err(panic) = executed {
            let error = TensorError::new(panic_message(panic.as_ref()));
            self.set_errors(&ordering, &mut context.handles, &error);
            self.did_not_run.extend(ordering.iter().copied());
            self.failed.get_or_insert(panic);
        }
    }

    /// Record `error` on the write sets of every operation at `ordering`.
    fn set_errors(
        &self,
        ordering: &[usize],
        handles: &mut HandleContainer<R::FusionHandle>,
        error: &TensorError,
    ) {
        for op in ordering.iter().map(|id| &self.ir[*id]) {
            set_output_errors(op, handles, error);
        }
    }

    pub(crate) fn execute_operations(
        &mut self,
        handles: &mut HandleContainer<R::FusionHandle>,
        ordering: &[usize],
    ) {
        self.num_executed += ordering.len();

        for id in ordering {
            let ir = &self.ir[*id];

            // A skip: an input this operation needs was never written, so it
            // does not run and its outputs take the same error — naming the
            // failure that started it rather than one of their own.
            let skip = input_error(ir, handles).map(TensorError::propagated);

            if let Some(error) = skip {
                set_output_errors(ir, handles, &error);
                self.did_not_run.push(*id);
                continue;
            }

            // Caught per operation, not per segment: a segment is just what
            // happened to be queued together, so a failure in one operation
            // says nothing about the next one unless they share a tensor — and
            // if they do, the next one skips on the error its input now
            // carries. Stopping the loop instead would make an unrelated
            // operation's outcome depend on queue order.
            let op = &self.operations[*id];
            let executed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op.execute(handles)));

            if let Err(panic) = executed {
                let error = TensorError::new(panic_message(panic.as_ref()));
                set_output_errors(ir, handles, &error);
                self.did_not_run.push(*id);
                self.failed.get_or_insert(panic);
            }
        }
    }
}
