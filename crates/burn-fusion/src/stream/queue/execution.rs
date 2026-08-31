use burn_ir::{HandleContainer, OperationIr, TensorError, TensorStatus};
use burn_std::config::{fusion::FusionLogLevel, log_fusion};
use std::sync::Arc;

use crate::{
    FusionRuntime, UnfusedOp,
    search::BlockOptimization,
    stream::{
        Context, ContextGuard, Executed, OperationConverter, OrderedExecution, RelativeOps,
        StreamId,
        execution::log_execution_table,
        panic_message,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
    },
};

use super::OperationQueue;

impl<R: FusionRuntime> OperationQueue<R> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<R::FusionHandle>,
        store: &mut ExecutionPlanStore<R::Optimization>,
        stream_id: StreamId,
    ) {
        let plan = store.get_mut_unchecked(id);

        // A cached plan may name relative shape ids this stream never assigned. Matching
        // on operations therefore does not imply the plan fits. When it does not, run the very
        // same operations in submission order instead: always a legal order, just unfused.
        //
        // The bound is what the plan's own operations assigned, not what the whole queue did:
        // plans usually fire from `ExecutionTrigger::OnOperations`, i.e. exactly when later
        // operations are already queued behind them, and those would otherwise inflate the
        // count enough to let an unfitting plan through.
        let len = plan.optimization.ordering.len();
        let assigned = match len.checked_sub(1).and_then(|i| self.shapes_assigned.get(i)) {
            Some(assigned) => *assigned,
            // No operation to bound against: nothing but shape id 0 can be legal.
            None => 1,
        };
        if let Some(max_id) = plan.optimization.strategy.max_relative_shape_id()
            && max_id >= assigned
        {
            log_fusion(FusionLogLevel::Medium, || {
                format!(
                    "[plan] #{id} needs relative shape id {max_id} but the stream assigned \
                     {assigned}; running its {len} operations unfused"
                )
            });

            let ordering: Vec<usize> = (0..len).collect();
            let mut fallback = BlockOptimization::new(
                ExecutionStrategy::Operations {
                    ordering: Arc::new(ordering.clone()),
                },
                ordering,
            );
            self.execute_block_optimization(&mut fallback, handles, stream_id);
            return;
        }

        self.execute_block_optimization(&mut plan.optimization, handles, stream_id);
    }

    /// Execute the queue with a one-off [`BlockOptimization`] that isn't stored in the cache.
    ///
    /// Used when fusion exploration is capped (see [`Explorer`](crate::stream::execution)): a
    /// cache-missing segment runs unfused without paying the optimizer cost or growing the store.
    pub(crate) fn execute_unfused(
        &mut self,
        mut optimization: BlockOptimization<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
        stream_id: StreamId,
    ) {
        self.execute_block_optimization(&mut optimization, handles, stream_id);
    }

    fn execute_block_optimization(
        &mut self,
        step: &mut BlockOptimization<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
        stream_id: StreamId,
    ) {
        log_execution_table(stream_id, &step.strategy, &self.global);

        let operations = core::mem::take(&mut self.operations);
        let ir = core::mem::take(&mut self.global);

        let executed = run_strategy(step, &mut self.converter, handles, operations, ir);
        let num_drained = executed.num_executed;

        // Restored before anything else looks at the queue. The strategy took
        // both lists by value, so an unwind that carried them away would leave
        // `relative` describing closures that no longer exist, and the next
        // plan to match would index into the gap.
        self.operations = executed.operations;
        self.global = executed.ir;

        if let Some(panic) = executed.failed {
            // Every failure's report is the claim it left on the tensors it
            // was going to write, which is delivered when one of them is read.
            // Logged here as the backstop for the one nobody ever reads.
            log::warn!(
                "a fused operation failed: {}; the tensors it was going to write are claimed, \
                 and reading one of them reports it",
                panic_message(panic.as_ref()),
            );
        }

        self.drain_queue(num_drained, handles);
    }

    /// Bookkeeping after executing `num_drained` operations from the queue.
    fn drain_queue(&mut self, num_drained: usize, handles: &mut HandleContainer<R::FusionHandle>) {
        self.global[0..num_drained]
            .iter()
            .flat_map(|desc| desc.nodes())
            .for_each(|tensor| {
                if tensor.status == TensorStatus::ReadWrite {
                    self.variables.remove(&tensor.id);
                };
                R::free_handle(handles, tensor)
            });

        self.global.drain(0..num_drained);

        self.reset_relative();
        // An execution boundary: release frees whose references just ran.
        self.flush_deferred(handles);
    }

    fn reset_relative(&mut self) {
        self.relative.clear();
        self.shapes_assigned.clear();
        self.converter.clear();

        for node in self.global.iter() {
            let relative = node.to_relative(&mut self.converter);
            self.relative.push(relative);
            self.shapes_assigned
                .push(self.converter.num_relative_shapes());
        }
    }
}

/// Drive one block's execution strategy.
///
/// Wraps the converter's per-block fields and the handle container into a single owned
/// [`Context`] via [`ContextGuard`] for the duration of this call, then threads `&mut Context`
/// through the recursive strategy walk. Operations-only strategies just grab
/// `&mut ctx.handles`; optimization strategies hand `&mut ctx` to the fused op.
fn run_strategy<R: FusionRuntime>(
    optimization: &mut BlockOptimization<R::Optimization>,
    converter: &mut OperationConverter,
    handles: &mut HandleContainer<R::FusionHandle>,
    operations: Vec<UnfusedOp<R>>,
    ir: Vec<OperationIr>,
) -> Executed<R> {
    let mut execution = OrderedExecution::new(operations, ir);
    let escaped = {
        let mut guard = ContextGuard::new(converter, handles);
        // A backstop. Each unit of work catches its own panic and claims its
        // own write set, so nothing should unwind this far — but if something
        // in the strategy walk itself does, this is the only frame that still
        // owns `execution`, and therefore the only one that can hand the
        // untouched lists back to the queue instead of dropping them
        // mid-unwind.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_strategy::<R>(&mut optimization.strategy, &mut guard, &mut execution);
        }))
        .err()
    };
    let mut executed = execution.finish();

    if let Some(escaped) = escaped {
        // Nothing says which operation it came from, so the whole consumed
        // segment is claimed — conservatively, leaving alone any output an
        // operation did write, so one failure does not turn into several.
        let error = TensorError::new(panic_message(escaped.as_ref()));
        for op in executed.ir.iter().take(executed.num_executed) {
            for node in op.outputs() {
                handles.claim_unwritten(node.id, error.clone());
            }
        }
        executed.failed.get_or_insert(escaped);
    }

    executed
}

fn execute_strategy<R: FusionRuntime>(
    strategy: &mut ExecutionStrategy<R::Optimization>,
    context: &mut Context<R::FusionHandle>,
    execution: &mut OrderedExecution<R>,
) {
    match strategy {
        ExecutionStrategy::Optimization { ordering, opt, .. } => {
            execution.execute_optimization(opt, context, ordering.clone());
        }
        ExecutionStrategy::Operations { ordering } => {
            execution.execute_operations(&mut context.handles, ordering);
        }
        ExecutionStrategy::Composed(items) => {
            for item in items.iter_mut() {
                execute_strategy::<R>(item, context, execution);
            }
        }
    }
}
