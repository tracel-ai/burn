use burn_ir::HandleContainer;

use crate::{
    FusionRuntime,
    search::BlockOptimization,
    stream::{
        Context, OperationConverter, OperationQueue, RelativeOps,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
    },
};

use super::OrderedExecution;

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

/// General trait to abstract how a single operation is executed.
pub trait Operation<R: FusionRuntime>: Send + Sync {
    /// Execute the operation.
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>);
}

enum Execution<'a, R: FusionRuntime> {
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

impl<'a, R: FusionRuntime> Execution<'a, R> {
    fn execute(
        optimization: &mut BlockOptimization<R::Optimization>,
        converter: &'a mut OperationConverter,
        handles: &'a mut HandleContainer<R::FusionHandle>,
        operations: Vec<Box<dyn Operation<R>>>,
    ) -> (Vec<Box<dyn Operation<R>>>, usize) {
        let ordering = optimization.ordering.clone();
        let execution = OrderedExecution::new(operations, ordering);

        if matches!(&optimization.strategy, ExecutionStrategy::Composed(..)) {
            let mut context = converter.context(handles);
            let mut this = Execution::Multiple {
                context: &mut context,
                execution,
            };

            this = this.execute_strategy(&mut optimization.strategy);

            match this {
                Execution::Multiple { execution, .. } => execution.finish(),
                _ => unreachable!(),
            }
        } else {
            let mut this = Execution::Single {
                handles,
                converter,
                execution,
            };
            this = this.execute_strategy(&mut optimization.strategy);

            match this {
                Execution::Single { execution, .. } => execution.finish(),
                _ => unreachable!(),
            }
        }
    }

    fn execute_strategy(mut self, strategy: &mut ExecutionStrategy<R::Optimization>) -> Self {
        match &mut self {
            Execution::Single {
                handles,
                converter,
                execution,
            } => match strategy {
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    let mut context = converter.context(handles);
                    execution.execute_optimization_with_fallbacks(opt, &mut context, fallbacks)
                }
                ExecutionStrategy::Optimization(opt) => {
                    let mut context = converter.context(handles);
                    execution.execute_optimization(opt, &mut context)
                }
                ExecutionStrategy::Operations(size) => execution.execute_operations(handles, *size),
                ExecutionStrategy::Composed(_) => unreachable!(),
            },
            Execution::Multiple { context, execution } => match strategy {
                ExecutionStrategy::Optimization(opt) => {
                    execution.execute_optimization(opt, context);
                }
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    execution.execute_optimization_with_fallbacks(opt, context, fallbacks);
                }
                ExecutionStrategy::Operations(size) => {
                    execution.execute_operations(&mut context.handles, *size);
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
            Execution::execute(step, &mut self.converter, handles, operations);

        self.operations = operations;
        self.drain_queue(num_drained, handles);
    }

    /// Bookkeeping after executing `num_drained` operations from the queue.
    fn drain_queue(&mut self, num_drained: usize, handles: &mut HandleContainer<R::FusionHandle>) {
        self.global[0..num_drained]
            .iter()
            .flat_map(|desc| desc.nodes())
            .for_each(|tensor| handles.free(tensor));

        self.global.drain(0..num_drained);
        self.reset_relative();
        assert_eq!(self.global.len(), self.operations.len());
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
