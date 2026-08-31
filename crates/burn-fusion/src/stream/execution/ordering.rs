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
    ordering: Option<Arc<Vec<usize>>>,
    /// The first panic a unit of this execution raised, kept only so the
    /// caller can log it. Every failure's report is the error it left on the
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
        // `min`: the count is taken before the work runs (see
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
            return;
        }

        let executed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            optimization.execute(context, self)
        }));

        if let Err(panic) = executed {
            let error = TensorError::new(panic_message(panic.as_ref()));
            self.set_errors(&ordering, &mut context.handles, &error);
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
                set_output_errors(&self.ir[*id], handles, &error);
                self.failed.get_or_insert(panic);
            }
        }
    }
}
