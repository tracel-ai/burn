use std::sync::{Arc, Mutex, OnceLock};

use burn_ir::{HandleContainer, OperationIr};

use super::{OnPanic, Outcome, Panic, WriteScope};

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
    pub(crate) failed: Option<Panic>,
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
    /// The same, for work that records it through a shared reference because it
    /// cannot hold one to this — a [`FallbackOp`], which outlives the borrow it
    /// was built from. Allocated on the first fallback and never otherwise, so
    /// a segment that uses none pays nothing; merged in [`finish`](Self::finish).
    deferred: OnceLock<Arc<Mutex<Vec<usize>>>>,
    ordering: Option<Arc<Vec<usize>>>,
    /// The first panic a unit of this execution raised, kept only so the
    /// caller can log it. Every failure's report is the error it left on the
    /// tensors, not this.
    failed: Option<Panic>,
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
    /// This operation's position in the segment, and where to record that it
    /// did not run. A skipped fallback was never replayed server-side, so a
    /// runtime whose handles live there has to hear about it or it strands the
    /// buffer — the same reason the unfused path records one.
    position: usize,
    did_not_run: Arc<Mutex<Vec<usize>>>,
}

impl<R: FusionRuntime> Clone for FallbackOp<R> {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
            ir: self.ir.clone(),
            position: self.position,
            did_not_run: self.did_not_run.clone(),
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
        let outcome = WriteScope::over(&self.ir, handles)
            .run(OnPanic::Raise, |handles| self.operation.execute(handles));

        match outcome {
            Outcome::Ran => {}
            // It did not write, so a runtime whose handles live on a server has
            // to hear about it. A panic does not arrive here — it is left to
            // the block's scope, which records the whole block.
            Outcome::Skipped | Outcome::Reported | Outcome::Panicked(_) => self
                .did_not_run
                .lock()
                .expect("no panic holds this lock")
                .push(self.position),
        }
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
                    position: index,
                    did_not_run: self.deferred().clone(),
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
            deferred: OnceLock::new(),
            ordering: None,
            failed: None,
        }
    }

    /// Where work that cannot reach `did_not_run` directly records a skip.
    fn deferred(&self) -> &Arc<Mutex<Vec<usize>>> {
        self.deferred.get_or_init(Default::default)
    }

    pub(crate) fn finish(mut self) -> Executed<R> {
        // `min`: the count is taken before the work runs (see
        // `execute_optimization`), so a strategy torn down by a panic can
        // leave it describing operations a shorter list never held.
        if let Some(deferred) = self.deferred.get() {
            let mut deferred = deferred.lock().expect("no panic holds this lock");
            self.did_not_run.append(&mut deferred);
        }

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
    /// Every unit of work counts its operations *before* it runs, so an escape
    /// normally leaves at least one consumed and the queue shrinks. One that
    /// consumes nothing is not a safe place to stop: the policy re-plans the
    /// identical queue, re-selects the same strategy and raises the same panic,
    /// without end — a silent hang, which is a worse outcome than the panic it
    /// replaced.
    ///
    /// No path is known to reach it: a plan that does not fit its segment is
    /// replaced before the walk begins, and every other failure happens inside
    /// a scope that counts first. It stays because the only argument for
    /// deleting it is that no panic escapes, which is the kind of claim that
    /// stops being true quietly — and it costs one comparison. Delete it when
    /// a panic can no longer cross this frame at all.
    ///
    /// `planned` is the block the strategy was going to run, consumed as one
    /// unit; nothing in it ran, so a claim on its write set is honest.
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
        self.ordering = Some(ordering.clone());
        let num_drained = optimization.len();
        // Counted before the call rather than after, so an unwind out of
        // `execute` still leaves these operations consumed. Counting after
        // would put them back on the queue for the next segment to retry,
        // re-running work whose inputs the torn-down execution already took —
        // and their outputs carry the failure instead, which is what a read
        // of one of them has to report.
        self.num_executed += num_drained;

        // One scope over the whole block, because a fused kernel is one unit of
        // work: it reads every input of every operation it replaced and writes
        // every output. One claimed input anywhere in it stops all of it, and a
        // failure anywhere in it leaves the whole write set unwritten.
        // Reborrowed: the optimization reads this execution while the scope
        // holds the context it writes through.
        let this = &*self;
        let outcome =
            WriteScope::over_block(&this.ir, &ordering, context).run(OnPanic::Catch, |context| {
                optimization.execute(context, this);
                Ok(())
            });

        match outcome {
            Outcome::Ran => {}
            Outcome::Skipped | Outcome::Reported => {
                self.did_not_run.extend(ordering.iter().copied())
            }
            Outcome::Panicked(panic) => {
                self.did_not_run.extend(ordering.iter().copied());
                self.failed.get_or_insert(panic);
            }
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
            let op = &self.operations[*id];

            // One scope per operation, not per segment: a segment is just what
            // happened to be queued together, so a failure in one operation
            // says nothing about the next one unless they share a tensor — and
            // if they do, the next one skips on the claim its input now
            // carries. Scoping the whole loop instead would make an unrelated
            // operation's outcome depend on queue order.
            let outcome =
                WriteScope::over(ir, handles).run(OnPanic::Catch, |handles| op.execute(handles));

            match outcome {
                Outcome::Ran => {}
                Outcome::Skipped | Outcome::Reported => self.did_not_run.push(*id),
                Outcome::Panicked(panic) => {
                    self.did_not_run.push(*id);
                    self.failed.get_or_insert(panic);
                }
            }
        }
    }
}
