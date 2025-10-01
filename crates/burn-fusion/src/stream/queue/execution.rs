use std::sync::Arc;

use burn_ir::{HandleContainer, TensorStatus};

use crate::{
    FusionRuntime,
    search::BlockOptimization,
    stream::{
        Context, Operation, OperationConverter, OrderedExecution, RelativeOps,
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
    ) {
        let plan = store.get_mut_unchecked(id);
        self.execute_block_optimization(&mut plan.optimization, handles);
    }

    fn execute_block_optimization(
        &mut self,
        step: &mut BlockOptimization<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let mut operations = Vec::new();
        core::mem::swap(&mut operations, &mut self.operations);
        let (operations, num_drained) =
            QueueExecution::run(step, &mut self.converter, handles, operations);

        self.operations = operations;
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
                handles.free(tensor)
            });

        self.global.drain(0..num_drained);

        self.reset_relative();
    }

    fn reset_relative(&mut self) {
        self.relative.clear();
        self.converter.clear();

        for node in self.global.iter() {
            let relative = node.to_relative(&mut self.converter);
            self.relative.push(relative);
        }
    }
}

/// A queue execution has the responsibility to run the provided
/// [optimization](FusionRuntime::Optimization) without holes.
enum QueueExecution<'a, R: FusionRuntime> {
    Single {
        handles: &'a mut HandleContainer<R::FusionHandle>,
        converter: &'a mut OperationConverter,
        execution: OrderedExecution<R>,
    },
    Multiple {
        context: &'a mut Context<'a, R::FusionHandle>,
        execution: OrderedExecution<R>,
    },
}

impl<'a, R: FusionRuntime> QueueExecution<'a, R> {
    fn run(
        optimization: &mut BlockOptimization<R::Optimization>,
        converter: &'a mut OperationConverter,
        handles: &'a mut HandleContainer<R::FusionHandle>,
        operations: Vec<Arc<dyn Operation<R>>>,
    ) -> (Vec<Arc<dyn Operation<R>>>, usize) {
        let execution = OrderedExecution::new(operations);

        if matches!(&optimization.strategy, ExecutionStrategy::Composed(..)) {
            let mut context = converter.context(handles);
            let mut this = QueueExecution::Multiple {
                context: &mut context,
                execution,
            };

            this = this.execute_strategy(&mut optimization.strategy);

            match this {
                QueueExecution::Multiple { execution, .. } => execution.finish(),
                _ => unreachable!(),
            }
        } else {
            let mut this = QueueExecution::Single {
                handles,
                converter,
                execution,
            };
            this = this.execute_strategy(&mut optimization.strategy);

            match this {
                QueueExecution::Single { execution, .. } => execution.finish(),
                _ => unreachable!(),
            }
        }
    }

    fn execute_strategy(mut self, strategy: &mut ExecutionStrategy<R::Optimization>) -> Self {
        match &mut self {
            QueueExecution::Single {
                handles,
                converter,
                execution,
            } => match strategy {
                ExecutionStrategy::Optimization { ordering, opt } => {
                    let mut context = converter.context(handles);
                    execution.execute_optimization(opt, &mut context, ordering.clone())
                }
                ExecutionStrategy::Operations { ordering } => {
                    execution.execute_operations(handles, ordering)
                }
                ExecutionStrategy::Composed(_) => unreachable!(),
            },
            QueueExecution::Multiple { context, execution } => match strategy {
                ExecutionStrategy::Optimization { opt, ordering } => {
                    execution.execute_optimization(opt, context, ordering.clone());
                }
                ExecutionStrategy::Operations { ordering } => {
                    execution.execute_operations(context.handles, ordering);
                }
                ExecutionStrategy::Composed(items) => {
                    for item in items.iter_mut() {
                        self = self.execute_strategy(item);
                    }
                }
            },
        };
        self
    }
}
