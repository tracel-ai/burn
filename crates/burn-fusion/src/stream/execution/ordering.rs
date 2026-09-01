use std::sync::{Arc, Mutex, OnceLock};

use burn_ir::{HandleContainer, OperationIr, TensorError};

use super::{OnPanic, Outcome, Panic, WriteScope, claim_block};

use crate::{FusionRuntime, NumOperations, Optimization, UnfusedOp, stream::Context};

/// What a finished [`OrderedExecution`] hands back to the queue.
pub(crate) struct Executed<R: FusionRuntime> {
    /// The operations it did not consume, to go back on the queue.
    pub operations: Vec<UnfusedOp<R>>,
    /// The segment's IR, likewise.
    pub ir: Vec<OperationIr>,
    /// How many operations it consumed, run or errored.
    pub num_executed: usize,
    /// Which of those consumed operations never ran — skipped on an errored
    /// input, or torn down by a panic. Indices into `ir`. Empty while nothing
    /// has failed, so carrying it allocates nothing on the common path.
    pub did_not_run: Vec<usize>,
    /// The first panic raised, kept only so the caller can log it — every
    /// failure's report is the error it left on the tensors.
    pub failed: Option<Panic>,
}

/// What work inside a fused block reports back to the execution running it.
///
/// A [`FallbackOp`] outlives the borrow it was built from, so it cannot reach
/// the [`OrderedExecution`] directly. Both things it has to say travel through
/// one shared slot rather than two, because they are said together: a fallback
/// that did not run is both an operation the server never replayed and a
/// failure of the block it sits in.
#[derive(Default)]
struct FallbackReports {
    /// Positions, into the segment's IR, of fallbacks that did not run. Merged
    /// into [`OrderedExecution::did_not_run`] by [`finish`](OrderedExecution::finish).
    did_not_run: Vec<usize>,
    /// The first failure a fallback reported or skipped on, if any.
    ///
    /// The first rather than the last: a later one is either the same failure
    /// arriving again through a tensor this one claimed, or a consequence of
    /// running on bytes it never wrote. Taken by
    /// [`execute_optimization`](OrderedExecution::execute_optimization), so a
    /// [`Composed`](crate::stream::store::ExecutionStrategy::Composed) strategy
    /// cannot carry one block's failure into the next.
    failure: Option<TensorError>,
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
    /// What work inside a fused block reported back. Allocated on the first
    /// fallback and never otherwise, so a segment that uses none pays nothing.
    reports: OnceLock<Arc<Mutex<FallbackReports>>>,
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
    /// This operation's position in the segment, and where to report back to.
    /// A fallback that did not run was never replayed server-side, so a runtime
    /// whose handles live there has to hear about it or it strands the buffer —
    /// the same reason the unfused path records one.
    position: usize,
    reports: Arc<Mutex<FallbackReports>>,
}

impl<R: FusionRuntime> Clone for FallbackOp<R> {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
            ir: self.ir.clone(),
            position: self.position,
            reports: self.reports.clone(),
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
    ///
    /// A fallback that skips or reports gets there by a different road to the
    /// same place. Both return normally, so neither stops the optimization —
    /// it goes on launching the kernels around this operation, and they write
    /// their outputs from bytes it never produced. So the failure is reported
    /// back to the execution, which claims the block's whole write set once the
    /// optimization returns. Nothing is claimed twice that matters: the claim
    /// carries this same failure, so a read below any of it names one cause.
    pub fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        let outcome = WriteScope::over(&self.ir, handles)
            .run(OnPanic::Raise, |handles| self.operation.execute(handles));

        let failure = match outcome {
            Outcome::Ran => return,
            Outcome::Skipped(error) | Outcome::Reported(error) => Some(error),
            // Not reached: this scope raises, so a panic is left to the block's
            // scope, which records the whole block. Handled rather than
            // unreachable, because being wrong about that would strand a
            // buffer rather than say so.
            Outcome::Panicked(_) => None,
        };

        // It did not write, so a runtime whose handles live on a server has to
        // hear about it.
        let mut reports = self.reports.lock().expect("no panic holds this lock");
        reports.did_not_run.push(self.position);
        if let Some(failure) = failure {
            reports.failure.get_or_insert(failure);
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
                    reports: self.reports().clone(),
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
            reports: OnceLock::new(),
            ordering: None,
            failed: None,
        }
    }

    /// Where work that cannot reach this execution directly reports back.
    fn reports(&self) -> &Arc<Mutex<FallbackReports>> {
        self.reports.get_or_init(Default::default)
    }

    /// The failure a fallback of the block just run reported, taken so no later
    /// block sees it.
    fn fallback_failure(&self) -> Option<TensorError> {
        self.reports
            .get()?
            .lock()
            .expect("no panic holds this lock")
            .failure
            .take()
    }

    pub(crate) fn finish(mut self) -> Executed<R> {
        // `min`: the count is taken before the work runs (see
        // `execute_optimization`), so a strategy torn down by a panic can
        // leave it describing operations a shorter list never held.
        if let Some(reports) = self.reports.get() {
            let mut reports = reports.lock().expect("no panic holds this lock");
            self.did_not_run.append(&mut reports.did_not_run);
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
            Outcome::Skipped(_) | Outcome::Reported(_) => {
                self.did_not_run.extend(ordering.iter().copied())
            }
            Outcome::Panicked(panic) => {
                self.did_not_run.extend(ordering.iter().copied());
                self.failed.get_or_insert(panic);
            }
        }

        // A fallback that did not run is a failure of the block around it, and
        // one the scope above cannot have seen: the optimization returned
        // normally, so `Outcome::Ran` is what it reports. Claiming here says
        // what that scope would have said had the failure reached it.
        //
        // It displaces on purpose, including over a panic the scope just
        // claimed with: the fallback's failure came first, and a panic raised
        // after it is either that same failure arriving through a tensor it
        // claimed or a consequence of running without one. The first cause is
        // the one a read should name.
        if let Some(failure) = self.fallback_failure() {
            claim_block(&self.ir, &ordering, &mut context.handles, &failure);
            self.did_not_run.extend(ordering.iter().copied());
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
                Outcome::Skipped(_) | Outcome::Reported(_) => self.did_not_run.push(*id),
                Outcome::Panicked(panic) => {
                    self.did_not_run.push(*id);
                    self.failed.get_or_insert(panic);
                }
            }
        }
    }
}
